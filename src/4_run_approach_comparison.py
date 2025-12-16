import json
import os
import pathlib
from argparse import ArgumentParser
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import copy
import wandb

from ..utils.logger import get_logger
from .vectorstore import FAISSVectorStore
from ..data import collate_fn
from ..data.datasets import polyvore
from ..evaluation.metrics import compute_cir_scores
from ..models.load import load_model
from ..utils.utils import seed_everything
from ..data.datatypes import FashionCompatibilityQuery, FashionComplementaryQuery
from . import score_candidates_batch

from collections import defaultdict

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
RESULT_DIR = SRC_DIR / 'results'
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

TOP_K = 15
MAX_ITERS = 3
DELTA_REL = 0.0001
DELTA_ABS = 1e-5
ITEM_PER_SEARCH = 8
POINT_TARGET = (6, 8)

POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR = "{polyvore_dir}/precomputed_rec_embeddings"
POLYVORE_PRECOMPUTED_CLIP_EMBEDDING_DIR = "{polyvore_dir}/precomputed_clip_embeddings"

POLYVORE_CATEGORIES = [
    'all-body', 'bottoms', 'tops', 'outerwear', 'bags', 
    'shoes', 'accessories', 'scarves', 'hats', 
    'sunglasses', 'jewellery', 'unknown'
]
CORE_CATS = {'all-body', 'tops', 'bottoms'}
ACCESSORY_CATS = {'outerwear', 'bags', 'scarves', 'hats', 'sunglasses', 'jewellery', 'shoes', 'accessories', 'unknown'}

STATEMENT_KEYWORDS = {'pattern', 'print', 'bold', 'statement'}

CP_MODEL_PATH = "./checkpoints/compatibility_clip_best.pth"
CIR_MODEL_PATH = "./checkpoints/complementary_clip_best.pth"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint')
    parser.add_argument('--batch_sz_per_gpu', type=int,
                        default=512)
    parser.add_argument('--n_workers_per_gpu', type=int,
                        default=4)
    parser.add_argument('--wandb_key', type=str, 
                        default=None)
    parser.add_argument('--seed', type=int, 
                        default=42)
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    parser.add_argument('--demo', action='store_true')
    
    return parser.parse_args()

def setup_dataloaders(args, logger):
    metadata = polyvore.load_metadata(args.polyvore_dir)
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)
    items = polyvore.PolyvoreItemDataset(args.polyvore_dir, metadata=metadata, load_image=False, embedding_dict=embedding_dict)
    logger.info(f"Loaded {len(items)} items from Polyvore {args.polyvore_type} dataset.")
    logger.info(f"Shape: {items[0].embedding.shape}")
    logger.info(f"Example: {items[0]}")

    test_triplet = polyvore.PolyvoreTripletDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
        dataset_split='test', metadata=metadata, load_image=False, embedding_dict=embedding_dict
    )
    test_triplet_dataloader = torch.utils.data.DataLoader(
        dataset=test_triplet, batch_size=args.batch_sz_per_gpu, shuffle=True,
        num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.triplet_collate_fn
    )

    test_fitb = polyvore.PolyvoreFillInTheBlankDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type,
        dataset_split='test', metadata=metadata, load_image=False, embedding_dict=embedding_dict
    )
    test_fitb_dataloader = torch.utils.data.DataLoader(
        dataset=test_fitb, batch_size=args.batch_sz_per_gpu, shuffle=False,
        num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.fitb_collate_fn
    )
    return metadata, embedding_dict, items, test_triplet, test_triplet_dataloader, test_fitb, test_fitb_dataloader

def assign_point(item, logger):
    if item is None:
        logger.warning(f"     ❌ Received None item {item}, skipping.")
        return 0
    
    desc = getattr(item, "description", "").lower()
    category = getattr(item, "category", "unknown").lower()
    if "pattern" in desc or "print" in desc or "bold" in desc or "statement" in desc:
        logger.info(f"     #️⃣ Item {item.item_id} with description: '{desc}' assigned 2 points for pattern/print/bold/statement in description of category {category}.")
        return 2
    # Statement shoes/jewelry/accessories by category descriptor
    if category in {"outerwear", "bags", "scarves", "hats", "sunglasses", "jewellery"}:
        if "bright" in desc or "chunky" in desc or "large" in desc:
            logger.info(f"     #️⃣ Item {item.item_id} with description: '{desc}' assigned 2 points for bright/chunky/large in description of category {category}.")
            return 2
    logger.info(f"     #️⃣ Item {item.item_id} with description: '{desc}' and category: '{category}' assigned 1 point by default.")
    return 1


def get_outfit_points(outfit, logger):
    points = sum(assign_point(item, logger) for item in outfit)
    cats = {item.category for item in outfit}
    if 'all-body' not in cats and ('tops' not in cats or 'bottoms' not in cats):
        points -= (1 if 'tops' not in cats else 0) + (1 if 'bottoms' not in cats else 0)
    return points


def search_best_candidate(query_emb, indexer, candidates, outfit, scorer, fill_model, logger, top_k=TOP_K, category=None):
    res = indexer.search(embeddings=query_emb, k=top_k, category=category)[0]
    
    # 🆕 Initialize defaults
    best_score = -float('inf')
    best_candidate = None
    
    unique_results = []
    seen_ids = set()
    for score, item_id in res:
        if item_id not in seen_ids and any(item.item_id == item_id for item in candidates):
            unique_results.append(item_id)
            seen_ids.add(item_id)

    logger.info(f"     🔍 Searching best candidate for category '{category}' among top-{top_k} results / {len(unique_results)} unique candidate(s).")

    # for candidate_id in unique_results:
    #     candidate_item = next((item for item in candidates if item.item_id == candidate_id), None)
    #     if candidate_item and candidate_item.item_id not in {item.item_id for item in outfit}:
    #         # Evaluate compatibility with the new outfit
    #         new_outfit = copy.deepcopy(outfit)
    #         new_outfit.append(candidate_item)
    #         new_query = FashionComplementaryQuery(outfit=new_outfit)
    #         score_cand = float(
    #             scorer.predict_score(query=[new_query], use_precomputed_embedding=True)[0].detach().cpu()
    #         )
    #         logger.info(f"        ➡️ Trying candidate {candidate_item.item_id} (category: {candidate_item.category}): score {score_cand:.4f}")

    #         if score_cand > best_score:
    #             best_score = score_cand
    #             best_candidate = candidate_item

    candidate_outfits = []
    candidate_items = []

    for candidate_id in unique_results:
        cand_item = next((item for item in candidates if item.item_id == candidate_id), None)
        if cand_item and cand_item.item_id not in {item.item_id for item in outfit}:
            new_outfit = list(outfit) + [cand_item]  # append candidate
            candidate_outfits.append(new_outfit)
            candidate_items.append(cand_item)

    if candidate_outfits:
        scores = score_candidates_batch.score_candidates_batch_items(
            candidate_outfits=candidate_outfits,
            model_transformer=fill_model,
            model_clip=scorer,
            batch_size=TOP_K
        )
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        best_candidate = candidate_items[best_idx]


    return best_candidate, best_score


def add_item_to_outfit(outfit, orig_points, missing_categories, candidates,
                      scorer, fill_model, indexer, logger,
                      trace, mode, point_max=8):
    
    used_item_ids = set(item.item_id for item in outfit)
    points = orig_points
    
    # 🆕 Initialize defaults BEFORE loop
    best_score = -float('inf')
    
    for cat in missing_categories:
        if mode == 'additional' and points >= point_max:
            break

        query = FashionComplementaryQuery(outfit=outfit, category=cat)
        query_emb = fill_model.embed_query(query=[query], use_precomputed_embedding=True).detach().cpu().numpy()
        
        best_cand_cat, best_score_cat = search_best_candidate(query_emb, indexer, candidates, outfit, scorer, fill_model, logger, top_k=TOP_K, category=cat)
        
        if best_cand_cat is None:
            logger.warning(f"     ❌ No candidate found for {cat}")
            continue
        
        if best_cand_cat.item_id not in used_item_ids:
            outfit.append(best_cand_cat)
            used_item_ids.add(best_cand_cat.item_id)
            points = get_outfit_points(outfit, logger)
            logger.info(f"   🟤 Added missing {mode} item {best_cand_cat.item_id} (category: {best_cand_cat.category}), new score: {best_score_cat:.4f}, new points: {points}")
            trace.append({
                'added_item': best_cand_cat.item_id,
                'category': best_cand_cat.category,
                'score': best_score_cat,
                'points': points,
                'points_change': points - orig_points
                })
            # 🆕 Track overall best across all categories
            if best_score_cat > best_score:
                best_score = best_score_cat
                best_candidate = best_cand_cat

    return outfit, best_score, points, trace


def addition_mode(outfit, orig_score, orig_points, candidates,
                  scorer, fill_model, indexer, logger,
                  point_min=6, point_max=8):
    present = [item.category for item in outfit]
    score = orig_score
    points = orig_points
    trace = []

    # ---- CORE LOGIC: Ensure all-body or (tops AND bottoms) ----
    has_all_body = 'all-body' in present
    has_top = 'tops' in present
    has_bottom = 'bottoms' in present

    missing_core = []
    if has_top and not has_bottom and not has_all_body:
        missing_core.append('bottoms')
    elif has_bottom and not has_top and not has_all_body:
        missing_core.append('tops')
    elif not has_top and not has_bottom and not has_all_body:
        missing_core.append('all-body')

    logger.info("     #️⃣ Missing core categories to consider for addition: " + ", ".join(missing_core))
    
    random.shuffle(missing_core)
    # Only suggest missing 'tops' or 'bottoms' if needed.
    outfit, score, points, trace = add_item_to_outfit(outfit, points, missing_core, candidates,
                                                           scorer, fill_model, indexer, logger,
                                                           trace, mode='core', point_max=point_max)

    # Refresh the present categories and continue as before
    present = [item.category for item in outfit]
    to_consider = [cat for cat in POLYVORE_CATEGORIES 
                   if cat not in present and cat not in {'tops', 'bottoms', 'all-body', 'unknown'}]
    to_consider.append('jewellery')
    logger.info("     #️⃣ Additional categories to consider: " + ", ".join(to_consider))

    random.shuffle(to_consider)

    outfit, score, points, trace = add_item_to_outfit(outfit, points, to_consider, candidates,
                                                           scorer, fill_model, indexer, logger,
                                                           trace, mode='additional', point_max=point_max)

    return True, outfit, score, points, trace


def compute_contributions(outfit_query, outfit, orig_score, scorer, logger):
    contributions = {}

    for idx, item in enumerate(outfit):
        if item.category in CORE_CATS:
            continue  # Skip core pieces
        reduced_outfit = copy.deepcopy(outfit)
        del reduced_outfit[idx]
        if len(reduced_outfit) < 2:  # Minimum outfit size
            continue
        reduced_query = copy.deepcopy(outfit_query)
        reduced_query.outfit = reduced_outfit
        score_wo = float(scorer.predict_score([reduced_query], use_precomputed_embedding=True)[0].detach().cpu())
        contributions[idx] = orig_score - score_wo
        logger.info(f"   #️⃣ Contribution of item {item.item_id} (idx {idx+1}/{len(outfit)}): {contributions[idx]:.4f} (score: {score_wo:.4f} without it)")
    return contributions


def remove_negative_item(outfit_query, outfit, orig_score, orig_points, scorer, logger, trace):
    """
    Find the least contributing item (negative or low contribution) and remove it.
    Returns a tuple indicating if removal happened, category of removed item, new outfit, new score, new points, updated trace.
    """
    contributions = compute_contributions(outfit_query, outfit, orig_score, scorer, logger)
    if not contributions:
        trace.append({'removal_mode_termination': 'no_bad_item_found_to_remove'})
        logger.warning("     ❌ No removable items found.")
        return False, None, outfit, orig_score, orig_points, trace

    # Identify item with the lowest contribution
    slot_to_remove = min(contributions, key=lambda i: contributions[i])
    negative_item = outfit[slot_to_remove]

    # Only remove if contribution is negative
    if contributions[slot_to_remove] >= 0:
        trace.append({'removal_mode_termination': 'no_negative_contribution_item_found_to_remove'})
        logger.warning("     ❌ No item with negative contribution found to remove.")
        return False, None, outfit, orig_score, orig_points, trace

    # Remove the item
    reduced_outfit = [item for idx, item in enumerate(outfit) if idx != slot_to_remove]
    new_query = FashionComplementaryQuery(outfit=reduced_outfit)
    new_score = float(scorer.predict_score([new_query], use_precomputed_embedding=True)[0].detach().cpu())
    new_points = get_outfit_points(reduced_outfit, logger)

    trace.append({
        'removed_item': negative_item.item_id,
        'category': negative_item.category,
        'removed_slot': slot_to_remove,
        'score': new_score,
        'points': new_points,
        'score_improvement': new_score - orig_score,
        'points_change': new_points - orig_points
    })

    logger.info(f"   🟤 Removed item {negative_item.item_id} (category: {negative_item.category}), new score: {new_score:.4f}, new points: {new_points}")
    return True, negative_item.category, reduced_outfit, new_score, new_points, trace


def removal_mode(outfit_query, outfit, orig_score, orig_points, candidates,
                    scorer, fill_model, indexer, logger,
                    item_per_search=ITEM_PER_SEARCH, top_k=TOP_K, delta_rel=DELTA_REL, delta_abs=DELTA_ABS):

    removal_trace = []
    removed, removed_category, new_reduced_outfit, new_score, new_points, removal_trace = remove_negative_item(outfit_query, outfit, orig_score, orig_points, scorer, logger, removal_trace)

    return removed, removed_category, new_reduced_outfit, new_score, new_points, removal_trace


def swap_at_slot(outfit_query, outfit, orig_score, orig_points, idx_to_swap, candidates, scorer, fill_model, indexer, logger,
                 top_k=TOP_K, delta_rel=DELTA_REL, delta_abs=DELTA_ABS):
    old_item = outfit[idx_to_swap]
    category = old_item.category
    trace = []

    # 1. Build FAISS query for that category, based on current outfit minus this item
    reduced_outfit = [item for j, item in enumerate(outfit) if j != idx_to_swap]
    swap_query = FashionComplementaryQuery(outfit=reduced_outfit, category=category)
    query_emb = fill_model.embed_query([swap_query], use_precomputed_embedding=True).detach().cpu().numpy()

    # 2. Retrieve top-k candidates in same category (excluding items already in outfit)
    faiss_res = indexer.search(query_emb, k=top_k, category=category)[0]
    outfit_ids = {item.item_id for item in outfit}
    candidate_items = []
    
    for score_faiss, item_id in faiss_res:
        if item_id in outfit_ids:
            continue
        cand = next((it for it in candidates if it.item_id == item_id), None)
        if cand is not None:
            candidate_items.append(cand)

    if not candidate_items:
        trace.append({'swapping_mode_termination': 'no_candidates_found_for_slot-based_swap'})
        logger.info("No candidates found for explicit slot-based swap.")
        return False, outfit, None, None, trace

    # 3. Score replacements for this slot only
    candidate_outfits = []
    best_score = orig_score
    best_item = None

    for cand in candidate_items:
        new_outfit = list(outfit)
        new_outfit[idx_to_swap] = cand
        candidate_outfits.append(new_outfit)

    if candidate_outfits:
        scores = score_candidates_batch.score_candidates_batch_items(
            candidate_outfits=candidate_outfits,  # Full item objects
            model_transformer=fill_model,
            model_clip=scorer,
            batch_size=len(candidate_outfits)  # Use actual batch size
        )
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        best_item = candidate_items[best_idx]
        
        # Fixed threshold logic
        threshold = orig_score + max(delta_rel * abs(orig_score), delta_abs)
        if best_score > threshold:
            outfit[idx_to_swap] = best_item
            new_points = get_outfit_points(outfit, logger)
            trace.append({
                "removed_item": old_item.item_id,
                "added_item": best_item.item_id,
                "score": best_score,
                "points": new_points,
                'score_improvement': best_score - orig_score,
                'points_change': new_points - orig_points
            })
            logger.info(f"✅ Slot swap: {old_item.item_id} → {best_item.item_id}, score {orig_score:.4f} → {best_score:.4f}")
            logger.info(f" 🔍 Threshold: {orig_score} + max({delta_rel}*{abs(orig_score):.4f}, {delta_abs}) = {orig_score + max(delta_rel * abs(orig_score), delta_abs):.4f}")
            return True, outfit, best_score, new_points, trace
        else:
            logger.info(f" 🔍 Best improvement: {best_score-orig_score:+.4f} < threshold {threshold-orig_score:+.4f}")

        logger.warning("❌ No beneficial replacement found for this slot.")
        trace.append({'swapping_mode_termination': 'no_beneficial_replacement_found_for_slot'})
        return False, outfit, float(orig_score), None, trace


def swapping_mode(outfit_query, outfit, orig_score, orig_points, candidates, scorer, fill_model, indexer, logger, **kwargs):
    trace = []
    
    # 1. Remove negative/least contributing item
    removed, removed_category, reduced_outfit, score_wo, points_wo, removal_trace = remove_negative_item(
        outfit_query, outfit, orig_score, orig_points, scorer, logger, trace=[]
    )
    
    if not removed or not removal_trace:
        trace.append({'removal_mode_s_termination': 'no_bad_item_found_to_remove'})
        return False, outfit, orig_score, orig_points, trace
    
    old_item_id = removal_trace[0]['removed_item']  # CORRECT KEY
    
    # 2. Try to add replacement
    add_trace = []
    new_outfit, new_score, new_points, add_trace = add_item_to_outfit(
        reduced_outfit, points_wo, [removed_category], candidates,
        scorer, fill_model, indexer, logger, add_trace, mode='replacement'
    )
    
    # 3. Check if addition actually happened
    if not add_trace or 'added_item' not in add_trace[0]:
        trace.append({'swapping_mode_termination': 'no_replacement_found_to_swap'})
        return False, outfit, orig_score, orig_points, trace
    
    new_item_id = add_trace[0]['added_item']
    
    # 4. Log success
    logger.info(f"✅ Replaced {old_item_id} → {new_item_id} | score {new_score:.4f} (plus {new_score-orig_score:+.4f}) | points {new_points} (from {orig_points})")
    
    trace.append({
        'removed_item': old_item_id,
        'added_item': new_item_id,
        'score': new_score,
        'points': new_points,
        'score_improvement': new_score - orig_score,
        'points_change': new_points - orig_points
    })
    
    return True, new_outfit, new_score, new_points, trace


def iterative_improve(initial_outfit_query, scorer, fill_model, indexer, logger, candidates,
                      max_iters=MAX_ITERS, top_k=TOP_K, delta_rel=DELTA_REL, delta_abs=DELTA_ABS):
    """Iterative improvement pipeline for outfit scoring and optimization."""
    trace = []

    # 1️⃣ Initialize
    outfit = copy.deepcopy(initial_outfit_query.outfit)
    score = float(
        scorer.predict_score([initial_outfit_query], use_precomputed_embedding=True)[0].detach().cpu()
    )
    points = get_outfit_points(outfit, logger)
    logger.info(f"🟠 Initial outfit: score={score:.4f}, points={points}, items={len(outfit)}")
    trace.append({'outfit': [item.item_id for item in outfit], 'score': score, 'points': points})

    iters = 0
    improved = True

    while improved and iters < max_iters:
        iters += 1
        improved = False
        logger.info(f"🔄 Iteration {iters}")

        # --- A. Addition mode (if points < 6) ---
        if points < 6:
            logger.info(f"   🔺 Points < 6 ({points}), running addition mode")
            added, outfit, score, points, add_trace = addition_mode(
                outfit, score, points, candidates,
                scorer=scorer, fill_model=fill_model, indexer=indexer, logger=logger
            )
            trace.extend(add_trace)
            if added:
                improved = True
                continue  # re-evaluate after addition

        # --- B. Removal mode (if points > 8) ---
        elif points > 8:
            logger.info(f"   🔻 Points > 8 ({points}), running removal mode")
            removed, removed_category, outfit, score, points, remove_trace = removal_mode(
                initial_outfit_query, outfit, score, points, candidates,
                scorer=scorer, fill_model=fill_model, indexer=indexer, logger=logger
            )
            trace.extend(remove_trace)
            if removed:
                improved = True
                continue  # re-evaluate after removal

        # --- C. Swapping mode (points in [6,8]) ---
        else:
            logger.info(f" ⚡ Points within target range ({points}), trying slot-based swap")
            
            contributions = compute_contributions(initial_outfit_query, outfit, score, scorer, logger)
            if contributions:
                slot_to_swap = min(contributions, key=lambda i: contributions[i])
                
                swapped, outfit, score_new, points_new, swap_trace = swap_at_slot(
                    initial_outfit_query, outfit, score, points, slot_to_swap, candidates, scorer, 
                    fill_model, indexer, logger, top_k=top_k
                )
                trace.extend(swap_trace)
                if swapped:
                    score = score_new
                    points = points_new
                    improved = True
                    continue  # re-evaluate after successful slot swap
            
            logger.info(" 🔄 Slot swap failed, trying remove+replace...")
            swapped, outfit, score_new, points_new, swap_trace = swapping_mode(
                initial_outfit_query, outfit, score, points, candidates,
                scorer=scorer, fill_model=fill_model, indexer=indexer, logger=logger,
                item_per_search=ITEM_PER_SEARCH, top_k=top_k, delta_rel=delta_rel, delta_abs=delta_abs
            )
            trace.extend(swap_trace)
            if swapped:
                score = score_new
                points = points_new
                improved = True
                continue  # re-evaluate after swap

        # else:
        #     logger.info(f"   ⚡ Points within target range ({points}), checking for negative items to swap")
        #     swapped, outfit, score_new, points_new, swap_trace = swapping_mode(
        #         initial_outfit_query, outfit, score, points, candidates,
        #         scorer=scorer, fill_model=fill_model, indexer=indexer, logger=logger,
        #         item_per_search=ITEM_PER_SEARCH, top_k=top_k, delta_rel=delta_rel, delta_abs=delta_abs
        #     )
        #     if swapped:
        #         trace.extend(swap_trace.get("replacement_trace", []))
        #         score = score_new
        #         points = points_new
        #         improved = True
        #         continue  # re-evaluate after swap

        # If none of the modes made improvements, break
        logger.info(f"   🔹 No more improvement found in this iteration")
        break

    logger.info(f"✅ Iterative improvement finished: final score={score:.4f}, points={points}, items={len(outfit)}")
    return [item.item_id for item in outfit], score, trace


def get_cir_scores(model, outfit, indexer):
    query = FashionComplementaryQuery(
        outfit=outfit,
        category='Unknown'
    )
    
    e = model.embed_query(
        query=[query],
        use_precomputed_embedding=True
    ).detach().cpu().numpy().tolist()
    
    res = indexer.search(
        embeddings=e,
        k=ITEM_PER_SEARCH
    )[0]

    return res

def process_validation(args):
    # Logging Setup
    project_name = f'approaches_comparison'
    logger = get_logger(project_name, LOGS_DIR)
    logger.info(f'Logger Setup Completed')

    # Load data
    metadata, embedding_dict, items, test_triplet, test_triplet_dataloader, test_fitb, test_fitb_dataloader = setup_dataloaders(args, logger)
    test_pbar = tqdm(test_fitb_dataloader, desc=f'[Test] Dataset')
    logger.info(f'Dataloaders Setup Completed')

    logger.info(f'Categories found in candidate pool: {POLYVORE_CATEGORIES}')

    # Load models
    cp_model = load_model(args.model_type, checkpoint=CP_MODEL_PATH)
    cp_model.eval()
    cir_model = load_model(args.model_type, checkpoint=CIR_MODEL_PATH)
    cir_model.eval()
    logger.info(f'Model Loaded')

    # Create the indexer and load saved indices
    rec_indexer = FAISSVectorStore(
        index_name='rec_index',
        d_embed=128,
        faiss_type='IndexFlatIP',
        base_dir=POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR.format(polyvore_dir=args.polyvore_dir),
        categories=POLYVORE_CATEGORIES
    )

    indexer = FAISSVectorStore(
        index_name='rec_index',
        d_embed=128,
        faiss_type='IndexFlatIP',
        base_dir=POLYVORE_PRECOMPUTED_REC_EMBEDDING_DIR.format(polyvore_dir=args.polyvore_dir),
    )

    # Initialize JSONL file for streaming results
    jsonl_path = os.path.join(RESULT_DIR, "iterative_fitb_results.jsonl")
    outputs_count = 0
    
    all_preds_s, all_labels_s = [], []

    # Iterate over test data
    with torch.no_grad():
        for i, data in enumerate(test_pbar):
            logger.info(f'Processing batch {i}: {len(data["query"])} queries')

            # Get initial partial outfit
            initial_partial_outfits = data['query']
            candidates = data['candidates']
            labels = data['label']

            # Exit loop after three batches if demo mode is on
            if args.demo and i > 2:
                break

            # --- Single-step FITB baseline ---
            batched_q_emb = cir_model(initial_partial_outfits, use_precomputed_embedding=True).unsqueeze(1)
            batched_c_embs = cir_model(sum(candidates, []), use_precomputed_embedding=True)
            batched_c_embs = batched_c_embs.view(-1, 4, batched_c_embs.shape[1])
            
            dists = torch.norm(batched_q_emb - batched_c_embs, dim=-1)
            preds = torch.argmin(dists, dim=-1)
            labels = torch.tensor(labels).cuda()
            scores = -dists.cpu().numpy()

            # Accumulate Results
            all_preds_s.append(preds.detach())
            all_labels_s.append(labels.detach())

            # Logging
            score = compute_cir_scores(all_preds_s[-1], all_labels_s[-1])
            logs = {**score}
            test_pbar.set_postfix(**logs)

            # Process and SAVE each outfit IMMEDIATELY
            for example_idx in range(len(initial_partial_outfits)):
                logger.info(f"👉 Starting iterative improvement process for outfit (number {example_idx+1} out of {len(initial_partial_outfits)}) ...")

                # --- Iterative FITB ---
                final_outfit, final_score, trace = iterative_improve(
                    initial_partial_outfits[example_idx],
                    scorer=cp_model, fill_model=cir_model,
                    indexer=rec_indexer, logger=logger,
                    candidates=items)
                
                single_step_final_outfit = [item.item_id for item in initial_partial_outfits[example_idx].outfit] + [candidates[example_idx][int(preds[example_idx])].item_id]
                single_step_outfit_items = [next(item for item in items if item.item_id == item_id) for item_id in single_step_final_outfit]
                single_step_query = FashionCompatibilityQuery(outfit=single_step_outfit_items)

                cir_final_outfit = get_cir_scores(cir_model, initial_partial_outfits[example_idx].outfit, indexer)

                log_item = {
                    "outfit_id": outputs_count,
                    "batch_id": i,
                    "example_idx": example_idx,
                    "initial": {
                        'outfit': [item.item_id for item in initial_partial_outfits[example_idx].outfit],
                        'removed_item': candidates[example_idx][data['label'][example_idx]].item_id,
                        'outfit_before_removal': [item.item_id for item in initial_partial_outfits[example_idx].outfit] + [candidates[example_idx][data['label'][example_idx]].item_id],
                        'score': float(cp_model.predict_score(query=[initial_partial_outfits[example_idx]], use_precomputed_embedding=True)[0].detach().cpu()),
                    },
                    "single_step": {
                        'final_outfit': single_step_final_outfit,
                        'final_score': float(cp_model.predict_score(query=[single_step_query], use_precomputed_embedding=True)[0].detach().cpu()),
                    },
                    "cir": {
                        "final_outfit": cir_final_outfit,
                        "final_score": float(cp_model.predict_score(query=[single_step_query], use_precomputed_embedding=True)[0].detach().cpu()),
                    },
                    "iterative": {
                        "final_outfit": final_outfit,
                        "final_score": final_score.item() if isinstance(final_score, torch.Tensor) else final_score,
                        "iterations": len(trace),
                        "trace": trace
                    },
                    "metadata": {
                        "dataset_split": "test",
                        "candidate_dataset": "train + valid",
                        "batch_score": {k: float(v) for k, v in score.items()}
                    }
                }

                # SAVE IMMEDIATELY to JSONL file
                with open(jsonl_path, 'a') as f:
                    f.write(json.dumps(log_item) + '\n')
                
                outputs_count += 1
                logger.info(f"✅ Saved outfit #{outputs_count} to {jsonl_path}")
            
        all_preds_s = torch.cat(all_preds_s).cuda()
        all_labels_s = torch.cat(all_labels_s).cuda()
        score = compute_cir_scores(all_preds_s, all_labels_s)
        logger.info(f"✅ [Test] Fill in the Blank --> {score}")
    
    logger.info(f"✅ Processing complete! {outputs_count} outfits saved to {jsonl_path}")

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    process_validation(args)