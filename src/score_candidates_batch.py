import torch
from torch.utils.data import DataLoader, Dataset
from ..data.datatypes import FashionComplementaryQuery

class CandidateOutfitDataset(Dataset):
    """
    Wraps candidate outfits into a dataset so they can be scored in batches.
    Assumes you already have a function to map item_ids → embeddings.
    """
    def __init__(self, candidate_outfits, embed_fn):
        self.outfits = candidate_outfits
        self.embed_fn = embed_fn

    def __len__(self):
        return len(self.outfits)

    def __getitem__(self, idx):
        outfit = self.outfits[idx]
        # outfit is [item_id1, item_id2, ...]
        item_embeds = [self.embed_fn(i) for i in outfit]  # (num_items, D)
        item_embeds = torch.stack(item_embeds)            # → tensor
        return item_embeds


def score_candidates_batch_items(candidate_outfits, model_transformer, model_clip, device="cuda", batch_size=32):
    all_scores = []
    
    for i in range(0, len(candidate_outfits), batch_size):
        batch_outfits = candidate_outfits[i:i+batch_size]
        
        # Convert to proper query format matching your working code
        batch_queries = []
        for outfit in batch_outfits:
            query = FashionComplementaryQuery(outfit=outfit)
            batch_queries.append(query)
        
        # Use the exact same model interface as your single-step baseline
        try:
            with torch.no_grad():
                q_embs = model_transformer(batch_queries, use_precomputed_embedding=True)  # (B, D)
                scores = model_clip(q_embs).squeeze().cpu().tolist()
                all_scores.extend(scores)
        except:
            # Fallback single scoring
            for query in batch_queries:
                score = float(model_clip.predict_score([query], use_precomputed_embedding=True)[0].detach().cpu())
                all_scores.append(score)
    
    return all_scores
