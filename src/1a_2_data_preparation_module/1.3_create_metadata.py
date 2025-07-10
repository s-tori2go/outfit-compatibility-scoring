import os
import json
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from src import config

for folder in config.image_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

def tags_to_prompt(tags):
    prompt = "A " + ", ".join(tags) + " outfit"
    return prompt

try:
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionCLIP')
    tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionCLIP')
    model = model.to(config.DEVICE)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load FashionCLIP model: {e}")

text_tokens = tokenizer(config.METADATA_TAGS).to(config.DEVICE)

def create_metadata_dir(img_dir, meta_target_dir):
    image_files = [f for f in os.listdir(img_dir)
                   if f.lower().endswith(tuple(config.IMAGE_FILE_EXTENSIONS))]

    for image_file in tqdm(image_files, desc=f"Processing {img_dir}"):
        image_path = os.path.join(img_dir, image_file)
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = preprocess_val(img).unsqueeze(0).to(config.DEVICE)

            with torch.no_grad(), torch.amp.autocast(device_type='mps'):
                image_features = model.encode_image(img_tensor, normalize=True)
                text_features = model.encode_text(text_tokens, normalize=True)
                probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).squeeze()

            topk = torch.topk(probs, k=3)
            top_indices = topk.indices.cpu().numpy()
            top_scores = topk.values.cpu().numpy()
            matched_tags = [config.METADATA_TAGS[i] for i in top_indices]
            weights = top_scores / top_scores.sum()  # Normalize to sum to 1

            base_name = os.path.splitext(image_file)[0]
            #out_path = os.path.join(meta_target_dir, f"{base_name[:-10]}.json")
            out_path = os.path.join(meta_target_dir, f"{base_name}.json")
            metadata = {
                "tags": matched_tags,
                "weights": weights.tolist(),
                "prompt": tags_to_prompt(matched_tags),
                "label": 1,  # Label 1 = positive outfits
            }
            with open(out_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to process {image_file}: {e}")

create_metadata_dir(config.TEST_DIR, config.METADATA_TEST_DIR)
#create_metadata_dir(config.SEGMENTED_POS_OUTFITS_DIR, config.METADATA_POS_OUTFITS_DIR)
#create_metadata_dir(config.SEGMENTED_NEG_OUTFITS_DIR, config.METADATA_NEG_OUTFITS_DIR)