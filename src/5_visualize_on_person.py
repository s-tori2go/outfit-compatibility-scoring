import pandas as pd
import json
import os
import pathlib
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
items_folder = SRC_DIR / "./datasets/polyvore/images"
metadata_path = SRC_DIR / "./datasets/polyvore/item_metadata.json"
mannequin_path = SRC_DIR / "./datasets/person.jpg"
output_folder = "./output_folder"
RESULT_DIR = SRC_DIR / 'results'
jsonl_path = RESULT_DIR / "./iterative_fitb_results.jsonl"

# ------------------------------------------------------------
# LOAD METADATA
# ------------------------------------------------------------

with open(metadata_path, 'r') as f:
    item_metadata = json.load(f)

item_to_category = {
    item['item_id']: item.get('semantic_category', 'other').lower()
    for item in item_metadata
}

LAYER_ORDER = [
    'shoes', 'bottoms', 'all-body', 'tops', 'outerwear', 'accessories',
    'bags', 'hats', 'sunglasses', 'jewellery', 'scarves', 'other'
]

POSITION_CONFIG = {
    'all-body':  {'x': 0.29, 'y': 0.17, 'width': 0.50, 'height': 0.80},
    'tops':      {'x': 0.29, 'y': 0.17, 'width': 0.46, 'height': 0.35},
    'bottoms':   {'x': 0.30, 'y': 0.48, 'width': 0.40, 'height': 0.35},
    'shoes':     {'x': 0.25, 'y': 0.87, 'width': 0.36, 'height': 0.18},

    'outerwear': {'x': 0.62, 'y': 0.18, 'width': 0.32, 'height': 0.55},
    'bags':      {'x': 0.02, 'y': 0.50, 'width': 0.22, 'height': 0.25},
    'scarves':   {'x': 0.68, 'y': 0.08, 'width': 0.22, 'height': 0.20},
    'hats':      {'x': 0.02, 'y': 0.05, 'width': 0.22, 'height': 0.18},
    'sunglasses':{'x': 0.70, 'y': 0.02, 'width': 0.18, 'height': 0.12},
    'jewellery': {'x': 0.02, 'y': 0.30, 'width': 0.22, 'height': 0.15},
    'accessories':{'x': 0.72, 'y': 0.55, 'width': 0.22, 'height': 0.22},

    'other':     {'x': 0.75, 'y': 0.30, 'width': 0.20, 'height': 0.25},
}

# BODY BOUNDING BOXES
BODY_BBOX = {
    "x": 0.27,
    "y": 0.16,
    "w": 0.52,
    "h": 0.88
}

TOP_BBOX = {
    "x": BODY_BBOX["x"],
    "y": BODY_BBOX["y"],
    "w": BODY_BBOX["w"],
    "h": BODY_BBOX["h"] * 0.45
}

BOTTOM_BBOX = {
    "x": BODY_BBOX["x"],
    "y": BODY_BBOX["y"] + BODY_BBOX["h"] * 0.40,
    "w": BODY_BBOX["w"],
    "h": BODY_BBOX["h"] * 0.55
}

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def get_category(item_id):
    return item_to_category.get(item_id, 'other')


def remove_background(img):
    img_array = np.array(img.convert('RGBA'))
    white_threshold = 240

    mask = np.all(img_array[:, :, :3] > white_threshold, axis=2)
    img_array[mask, 3] = 0

    return Image.fromarray(img_array, 'RGBA')


def get_scale_factor(category):
    if category == 'all-body':
        return 2.0
    if category == 'shoes':
        return 0.7
    return 1.0


# ------------------------------------------------------------
# RESIZE + POSITION (FIXED VERSION)
# ------------------------------------------------------------

def resize_and_position(img, canvas_size, config, scale_factor=1.0, category=None):
    canvas_w, canvas_h = canvas_size

    # On-body items → Use bounding boxes
    if category == "all-body":
        bb = BODY_BBOX
    elif category == "tops":
        bb = TOP_BBOX
    elif category == "bottoms":
        bb = BOTTOM_BBOX
    else:
        bb = None

    # --- BODY REGION POSITIONING ---
    if bb is not None:
        body_x = int(canvas_w * bb["x"])
        body_y = int(canvas_h * bb["y"])
        body_w = int(canvas_w * bb["w"])
        body_h = int(canvas_h * bb["h"])

        img_aspect = img.width / img.height

        target_h = int(body_h * scale_factor)
        target_w = int(target_h * img_aspect)

        if target_w > body_w:
            target_w = body_w
            target_h = int(target_w / img_aspect)

        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        x = body_x + (body_w - target_w) // 2
        y = body_y

        return img, (x, y)

    # --- DEFAULT POSITIONING ---
    target_w = int(canvas_w * config['width'] * scale_factor)
    target_h = int(canvas_h * config['height'] * scale_factor)

    img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

    x = int(canvas_w * config['x'] + (target_w - img.width) / 2)
    y = int(canvas_h * config['y'])

    return img, (x, y)


# ------------------------------------------------------------
# OVERLAY REMOVED ITEM
# ------------------------------------------------------------

def overlay_removed_item(canvas, removed_item_id, mannequin_path, items_folder):
    canvas_w, canvas_h = canvas.size

    cat = get_category(removed_item_id)
    cfg = POSITION_CONFIG.get(cat, POSITION_CONFIG['other'])

    item_path = os.path.join(items_folder, f"{removed_item_id}.jpg")
    if not os.path.exists(item_path):
        print(f"⚠️ Removed item {removed_item_id} not found.")
        return canvas

    item_img = Image.open(item_path)
    item_img = remove_background(item_img)

    scale = get_scale_factor(cat)
    item_img, (x, y) = resize_and_position(
        item_img, canvas.size, cfg, scale_factor=scale, category=cat
    )

    canvas.paste(item_img, (x, y), item_img)

    # Draw outline + label
    draw = ImageDraw.Draw(canvas)
    outline_w = 6
    draw.rectangle([x, y, x + item_img.width, y + item_img.height],
                   outline="red", width=outline_w)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 28)
    except:
        font = ImageFont.load_default()

    label = "Removed Item"
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = x + (item_img.width - text_w) // 2
    text_y = y + item_img.height + 5
    draw.text((text_x, text_y), label, fill="red", font=font)

    return canvas


# ------------------------------------------------------------
# MAIN OUTFIT COMPOSITOR
# ------------------------------------------------------------

def create_outfit_composite(
    item_ids,
    mannequin_path,
    items_folder,
    output_path,
    removed_item_id=None,
    add_removed_overlay=False
):
    base = Image.open(mannequin_path).convert('RGBA')
    canvas = base.copy()
    canvas_size = canvas.size

    items_by_category = {}
    for item_id in item_ids:
        cat = get_category(item_id)
        items_by_category.setdefault(cat, []).append(item_id)

    per_cat_index = {cat: 0 for cat in items_by_category.keys()}

    for layer_category in LAYER_ORDER:
        if layer_category not in items_by_category:
            continue

        for item_id in items_by_category[layer_category]:
            item_path = os.path.join(items_folder, f"{item_id}.jpg")
            if not os.path.exists(item_path):
                print(f"Warning: Item {item_id} not found")
                continue

            item_img = Image.open(item_path)
            item_img = remove_background(item_img)

            cfg = dict(POSITION_CONFIG.get(layer_category, POSITION_CONFIG["other"]))
            idx = per_cat_index[layer_category]

            # offset non-body duplicates
            if layer_category not in ["all-body", "tops", "bottoms", "shoes"]:
                cfg['x'] = min(0.95, cfg['x'] + 0.05 * (idx % 3))
                cfg['y'] = min(0.95, cfg['y'] + 0.18 * (idx // 3))

            per_cat_index[layer_category] += 1

            scale = get_scale_factor(layer_category)
            item_img, pos = resize_and_position(
                item_img, canvas_size, cfg, scale_factor=scale, category=layer_category
            )

            canvas.paste(item_img, pos, item_img)

    if add_removed_overlay and removed_item_id is not None:
        canvas = overlay_removed_item(canvas, removed_item_id, mannequin_path, items_folder)

    canvas.save(output_path)
    return canvas


# ------------------------------------------------------------
# COMPARISON IMAGE GENERATOR
# ------------------------------------------------------------

def create_comparison_image(item_ids_list, labels, output_path, removed_item_id=None):
    imgs = []
    canvas_width = 3300
    canvas_height = 1300
    img_width = canvas_width // len(item_ids_list)

    for i, (item_ids, label) in enumerate(zip(item_ids_list, labels)):
        # show removed item only on the initial image (index 0)
        is_initial = (i == 0)

        outfit_img = create_outfit_composite(
            item_ids,
            mannequin_path,
            items_folder,
            "temp.png",
            removed_item_id=removed_item_id if is_initial else None,
            add_removed_overlay=is_initial
        )
        outfit_img = outfit_img.resize((img_width - 20, canvas_height - 40),
                                       Image.Resampling.LANCZOS)
        imgs.append(outfit_img)

    comparison = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))

    for i, img in enumerate(imgs):
        x = i * img_width
        comparison.paste(img, (x + 10, 20), img)

        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        draw.text((x + 10, 5), labels[i], fill=(0, 0, 0), font=font)

    comparison.save(output_path)

    try:
        os.remove("temp.png")
    except:
        pass


# ------------------------------------------------------------
# PROCESS JSONL → COMPARISONS
# ------------------------------------------------------------

def process_comparisons(jsonl_path, max_outfits=None):
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_json(jsonl_path, lines=True)

    if max_outfits:
        df = df.head(max_outfits)

    print(f"Processing {len(df)} outfits...")

    for idx, row in df.iterrows():
        outfit_id = row.outfit_id

        initial_ids = row.initial['outfit']
        single_ids = row.single_step['final_outfit']
        iterative_ids = row.iterative['final_outfit']

        # --- important: extract the removed item id from the JSON row
        removed_id = row.initial.get('removed_item', None)

        item_ids_list = [initial_ids, single_ids, iterative_ids]

        labels = [
            f"Initial ({len(row.initial['outfit'])} items), score {row.initial['score']:.3f}",
            f"Single-step ({len(row.single_step['final_outfit'])} items), score {row.single_step['final_score']:.3f}",
            f"Iterative ({len(row.iterative['final_outfit'])} items), score {row.iterative['final_score']:.3f}"
        ]

        output_path = os.path.join(output_folder, f"comparison_{outfit_id:04d}.png")

        print(f" - Creating comparison {outfit_id} (removed={removed_id})...")
        create_comparison_image(item_ids_list, labels, output_path, removed_item_id=removed_id)

    print("\n✓ All comparisons saved!")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    process_comparisons(jsonl_path, max_outfits=35)
    print("\n🎉 Comparison visualization complete!")
