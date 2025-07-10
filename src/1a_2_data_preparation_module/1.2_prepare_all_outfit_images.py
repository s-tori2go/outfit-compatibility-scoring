import os
import time

import cv2
import numpy as np
import supervision as sv
from groundingdino.util.inference import Model
from rembg import remove
from tqdm import tqdm

from src import config
from src import helpers

start = time.time()

for folder in config.image_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 0.1 Download models if missing
helpers.download_file(config.GROUNDING_DINO_CONFIG_URL, config.GROUNDING_DINO_CONFIG_PATH)
helpers.download_file(config.GROUNDING_DINO_CHECKPOINT_URL, config.GROUNDING_DINO_CHECKPOINT_PATH)

print("Using device:", config.DEVICE)

# 0.3 Load Grounding DINO
try:
    grounding_dino_model = Model(
    model_config_path=config.GROUNDING_DINO_CONFIG_PATH,
    model_checkpoint_path=config.GROUNDING_DINO_CHECKPOINT_PATH,
    device=config.DEVICE
    )
    print("✅ Grounding DINO loaded successfully")
except Exception as e:
    print(f"❌ Grounding DINO loading failed: {str(e)}")
    if "No module named 'groundingdino'" in str(e):
        print("Installing GroundingDINO...")
        os.system("pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
        # Retry after install
        grounding_dino_model = Model(
        model_config_path=config.GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=config.GROUNDING_DINO_CHECKPOINT_PATH,
        device=config.DEVICE
        )

def prepare_images(original_img_dir, segmented_img_dir, cropped_img_dir):
    image_files = [
        f for f in os.listdir(original_img_dir)
        if f.lower().endswith(config.IMAGE_FILE_EXTENSIONS)
    ]

    if not image_files:
        raise ValueError("❌ No images found in the directory!")

    already_processed = {f.replace('_segmented.jpg', '.jpg') for f in os.listdir(segmented_img_dir)}
    to_process = [f for f in image_files if f not in already_processed]
    print(f"Found {len(image_files)} outfit images, {len(already_processed)} already processed and {len(to_process)} to process.")

    for image_file in tqdm(to_process, desc=f"Processing {original_img_dir}"):
        image = cv2.imread(os.path.join(original_img_dir, image_file))

        detections, phrases = grounding_dino_model.predict_with_caption(
            image=image,
            caption='a person wearing an outfit',
            box_threshold=config.BOX_THRESHOLD,
            text_threshold=config.TEXT_THRESHOLD
        )

        boxes = detections.xyxy
        H, W = image.shape[:2]
        margin = 30

        cropped_image = image.copy()

        for i, box in enumerate(boxes):
            if i == 0:  # Only process the first detected person
                x1 = max(0, int(box[0]) - margin)
                y1 = max(0, int(box[1]) - margin)
                x2 = min(W, int(box[2]) + margin)
                y2 = min(H, int(box[3]) + margin)
                cropped_image = image[y1:y2, x1:x2]
                image_path = f"{os.path.join(cropped_img_dir, image_file[:-4])}_{i}_cropped.jpg"
                cv2.imwrite(image_path, cropped_image)

        image_without_bg = remove(cropped_image)

        # Create white background
        height, width = image_without_bg.shape[:2]
        segmented_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        alpha_channel = image_without_bg[:, :, 3] / 255.0
        for c in range(3):  # R, G, B channels
            segmented_image[:, :, c] = (
                    image_without_bg[:, :, c] * alpha_channel +
                    segmented_image[:, :, c] * (1 - alpha_channel)
            ).astype(np.uint8)

        canvas_height, canvas_width = 1200, 1200
        if height > canvas_height or width > canvas_width:
            scale = min(canvas_height / height, canvas_width / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            segmented_image = cv2.resize(segmented_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            height, width = segmented_image.shape[:2]

        start_y = (canvas_height - height) // 2
        start_x = (canvas_width - width) // 2
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        canvas[start_y:start_y + height, start_x:start_x + width] = segmented_image

        # Save
        image_path = f"{os.path.join(segmented_img_dir, image_file[:-4])}_segmented.jpg"
        cv2.imwrite(image_path, canvas)

    print(f"⏱️ Took {time.time() - start:.2f}s")

prepare_images(
    config.ORIGINAL_POS_OUTFITS_DIR,
    config.SEGMENTED_POS_OUTFITS_DIR,
    config.CROPPED_POS_OUTFITS_DIR
)

prepare_images(
    config.ORIGINAL_NEG_OUTFITS_DIR,
    config.SEGMENTED_NEG_OUTFITS_DIR,
    config.CROPPED_NEG_OUTFITS_DIR
)