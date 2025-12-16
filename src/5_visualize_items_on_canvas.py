from PIL import Image
from PIL import ImageDraw, ImageFont
import json
import os
import pathlib

# Configuration
SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
items_folder = SRC_DIR / "./datasets/polyvore/images"
metadata_path = SRC_DIR / "./datasets/polyvore/item_metadata.json"
output_folder = "./output_folder"

# Load metadata
with open(metadata_path, 'r') as f:
    item_metadata = json.load(f)

BG_COLOR = (255, 255, 255, 255)
TEXT_COLOR = (40, 40, 40, 255)
CANVAS_W, CANVAS_H = 1200, 1000
GRID_COLS = 5  # More columns for single category
ITEM_SIZE = (220, 220)  # Uniform size since single category
MARGIN_X, MARGIN_Y = 30, 30
LABEL_HEIGHT = 50

# Create category -> item_ids mapping
category_items = {}
for item in item_metadata:
    cat = item.get('semantic_category', 'other').lower()
    item_id = item['item_id']
    if cat not in category_items:
        category_items[cat] = []
    category_items[cat].append(item_id)

print("Available categories:")
for cat, count in sorted([(k, len(v)) for k, v in category_items.items() if len(v) > 0]):
    print(f"  {cat}: {count} items")

def resize_into_box(img, box_size):
    w, h = img.size
    bw, bh = box_size
    scale = min(bw / w, bh / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def get_category(item_id):
    for item in item_metadata:
        if item['item_id'] == item_id:
            return item.get('semantic_category', 'other').lower()
    return 'other'

def create_item_label(draw, item_id, cat, paste_x, paste_y, img_height, cell_w):
    """Add category and item_id label below the image"""
    label_y = paste_y + img_height + 10
    simple_text = f"{cat.title()} (ID: {item_id})"
    #simple_text = f"ID: {item_id}"
    draw.text((paste_x + 10, label_y), simple_text, fill=TEXT_COLOR)

def create_category_canvas(category, item_ids, items_folder, output_path):
    canvas = Image.new('RGBA', (CANVAS_W, CANVAS_H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    
    # Add title
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except:
        title_font = ImageFont.load_default()
    
    title = f"30 Samples of {category.replace('_', ' ').title()} items"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    title_x = (CANVAS_W - title_w) // 2
    #draw.text((title_x, 20), title, fill=TEXT_COLOR, font=title_font)

    x = MARGIN_X
    y = 50  # Start below title
    col = 0
    cell_w = (CANVAS_W - MARGIN_X * 2) // GRID_COLS
    row_max_h = 0

    for item_id in item_ids:
        img_path = os.path.join(items_folder, f"{item_id}.jpg")
        if not os.path.exists(img_path):
            print(f"  Warning: missing {img_path}")
            continue

        img = Image.open(img_path).convert('RGBA')
        img = resize_into_box(img, ITEM_SIZE)

        # Center in cell
        cell_center_x = MARGIN_X + col * cell_w + cell_w // 2
        paste_x = int(cell_center_x - img.width / 2)
        paste_y = y

        # Bounds checking
        paste_x = int(max(MARGIN_X, min(paste_x, CANVAS_W - img.width - MARGIN_X)))
        paste_y = int(max(100, min(paste_y, CANVAS_H - img.height - MARGIN_Y - LABEL_HEIGHT)))

        # Paste image
        canvas.paste(img, (paste_x, paste_y), img)

        # Add item label
        cat = get_category(item_id)
        create_item_label(draw, item_id, cat, paste_x, paste_y, img.height, img.width)

        # Track row height
        total_item_h = img.height + LABEL_HEIGHT
        row_max_h = max(row_max_h, total_item_h)

        col += 1
        if col >= GRID_COLS:
            col = 0
            x = MARGIN_X
            y += row_max_h + MARGIN_Y * 1.2
            row_max_h = 0
        else:
            x += cell_w

    canvas.save(output_path)
    print(f"✅ Saved: {output_path}")
    return canvas

def showcase_category(category, num_items):
    """Showcase N items from a specific category"""
    os.makedirs(output_folder, exist_ok=True)
    
    if category not in category_items:
        print(f"❌ Category '{category}' not found!")
        print("Available categories listed above.")
        return
    
    # Get random sample or first N items
    all_items = category_items[category]
    item_ids = all_items[:num_items]  # Take first N for consistency
    
    print(f"\n🎨 Showcasing {len(item_ids)} '{category}' items...")
    
    output_path = os.path.join(output_folder, f"{category}_showcase.png")
    create_category_canvas(category, item_ids, items_folder, output_path)

if __name__ == "__main__":
    showcase_category(category='all-body', num_items=15)
    print("\n🎉 Category showcase complete! Check ./output_folder/")
