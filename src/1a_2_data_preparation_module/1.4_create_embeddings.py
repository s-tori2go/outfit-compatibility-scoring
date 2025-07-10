import json
import os

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from src import config

for folder in config.image_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

try:
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionCLIP')
    tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionCLIP')
    print("✅ FashionCLIP model and tokenizer loaded.")
    model = model.to(config.DEVICE)
    print(f"✅ Using device: {config.DEVICE}")
except Exception as e:
    print("❌ Failed to load FashionCLIP model:", e)

def embed_image(image_file, directory):
    """ Embed an image using FashionCLIP model. """
    try:
        img = Image.open(os.path.join(directory, image_file)).convert("RGB")
        img_tensor = preprocess_val(img).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad(), torch.amp.autocast(device_type='mps'):
            embedding = model.encode_image(img_tensor, normalize=False).cpu().numpy()
        return embedding
    except Exception as e:
        print(f"❌ Failed to embed {image_file}: {e}")
        return None

def embed_text(text_content):
    """Embed text content using FashionCLIP model."""
    try:
        tokens = tokenizer(text_content).to(config.DEVICE)
        with torch.no_grad(), torch.amp.autocast(device_type="mps"):
            embedding = model.encode_text(tokens, normalize=False).cpu().numpy()
        return embedding
    except Exception as e:
        print(f"❌ Failed to embed text: {e}")
        return None

def save_embedding(embedding, base_name, target_directory):
    """ Save the embedding to a .npy file. """
    try:
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        embeddings_path = f"{os.path.join(target_directory, base_name)}.npy"
        np.save(embeddings_path, embedding.astype(np.float32))
    except Exception as e:
        print(f"❌ Failed to save embedding for {base_name}: {e}")

def process_outfits(img_dir, metadata_dir, img_target_dir, meta_target_dir):
    """Process images and metadata with progress tracking"""
    image_files = [f for f in os.listdir(img_dir)
                   if f.lower().endswith(config.IMAGE_FILE_EXTENSIONS)]

    for img_file in tqdm(image_files, desc=f"Processing {img_dir}"):
        base_name = os.path.splitext(img_file)[0][:-10]
        embeded_image = embed_image(img_file, img_dir)
        save_embedding(embeded_image, base_name, img_target_dir)

        # Embed metadata
        meta_path = os.path.join(metadata_dir, f"{base_name}.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            caption_embed = embed_text(metadata['prompt'])
            np.save(os.path.join(meta_target_dir, f"{base_name}_caption.npy"), caption_embed)

            tags = metadata['tags']
            if tags:
                tag_embeds = embed_text(tags)
                np.save(os.path.join(meta_target_dir, f"{base_name}_tags.npy"), tag_embeds)
                np.save(os.path.join(meta_target_dir, f"{base_name}_weights.npy"), np.array(metadata['weights']))
            else:
                # Handle empty tags case
                np.save(os.path.join(meta_target_dir, f"{base_name}_tags.npy"), np.zeros((0, 512)))
                np.save(os.path.join(meta_target_dir, f"{base_name}_weights.npy"), np.zeros(0))

process_outfits(
    config.SEGMENTED_POS_OUTFITS_DIR,
    config.METADATA_POS_OUTFITS_DIR,
    config.EMBEDDINGS_SEGMENTED_POS_OUTFITS_DIR,
    config.EMBEDDINGS_METADATA_POS_OUTFITS_DIR
)

process_outfits(
    config.SEGMENTED_NEG_OUTFITS_DIR,
    config.METADATA_NEG_OUTFITS_DIR,
    config.EMBEDDINGS_SEGMENTED_NEG_OUTFITS_DIR,
    config.EMBEDDINGS_METADATA_NEG_OUTFITS_DIR
)