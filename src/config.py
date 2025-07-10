import torch

IMAGE_FILE_EXTENSIONS = (".jpg", ".jpeg", ".png")

ORIGINAL_POS_OUTFITS_DIR = "../../data/outfits/positive_outfits/original_images"
CROPPED_POS_OUTFITS_DIR = "../../data/outfits/positive_outfits/cropped"
SEGMENTED_POS_OUTFITS_DIR = "../../data/outfits/positive_outfits/segmented"
EMBEDDINGS_SEGMENTED_POS_OUTFITS_DIR = "../../data/outfits/positive_outfits/embeddings"
METADATA_POS_OUTFITS_DIR = "../../data/outfits/positive_outfits/metadata"
EMBEDDINGS_METADATA_POS_OUTFITS_DIR = "../../data/outfits/positive_outfits/metadata_embeddings"

ORIGINAL_NEG_OUTFITS_DIR = "../../data/outfits/negative_outfits/original_images"
CROPPED_NEG_OUTFITS_DIR = "../../data/outfits/negative_outfits/cropped"
SEGMENTED_NEG_OUTFITS_DIR = "../../data/outfits/negative_outfits/segmented"
EMBEDDINGS_SEGMENTED_NEG_OUTFITS_DIR = "../../data/outfits/negative_outfits/embeddings"
METADATA_NEG_OUTFITS_DIR = "../../data/outfits/negative_outfits/metadata"
EMBEDDINGS_METADATA_NEG_OUTFITS_DIR = "../../data/outfits/negative_outfits/metadata_embeddings"

TEST_DIR = "../../data/outfits/outfits_to_test"
METADATA_TEST_DIR = "../../data/outfits/metadata_test"

EMBEDDINGS_DIR = "../../data/embeddings"
ITEM_EMBEDDINGS_DIR = "../../data/embeddings/item_embeddings"
COMBINED_EMBEDDINGS_DIR = "../../data/embeddings/combined_embeddings"
COMPATIBILITY_MODELS_DIR = "../../models/compatibility"

image_folders = [
    ORIGINAL_POS_OUTFITS_DIR,
    SEGMENTED_POS_OUTFITS_DIR,
    CROPPED_POS_OUTFITS_DIR,
    METADATA_POS_OUTFITS_DIR,
    EMBEDDINGS_METADATA_POS_OUTFITS_DIR,
    EMBEDDINGS_SEGMENTED_POS_OUTFITS_DIR,
    ORIGINAL_NEG_OUTFITS_DIR,
    SEGMENTED_NEG_OUTFITS_DIR,
    CROPPED_NEG_OUTFITS_DIR,
    METADATA_NEG_OUTFITS_DIR,
    EMBEDDINGS_METADATA_NEG_OUTFITS_DIR,
    EMBEDDINGS_SEGMENTED_NEG_OUTFITS_DIR,
    TEST_DIR,
    METADATA_TEST_DIR
]

SEGMENT_ANYTHING_MODEL_TYPE = "vit_h"
SEGMENT_ANYTHING_CHECKPOINT_PATH = "../../models/segmentanything/sam_vit_h.pth"
SEGMENT_ANYTHING_DOWNLOAD_URL = "https://hf-mirror.com/HCMUE-Research/SAM-vit-h/resolve/main/sam_vit_h_4b8939.pth?download=true"

GROUNDING_DINO_CONFIG_PATH = "../../models/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "../../models/groundingdino/weights/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG_URL = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

OBJ_DETECTION_ONNX_MODEL_PATH = "../../models/yolo/best.onnx"
OBJ_DETECTION_DATA_PATH = "../../models/yolo/data.yaml"

FASHION_CLIP_MODEL_TYPE = 'fashion_clip'
FASHION_CLIP_MODEL_PATH = "../../models/fashionclip/fashion-clip.pt"
FASHION_CLIP_DOWNLOAD_URL = "https://huggingface.co/s-nlp/fashion-clip/resolve/main/fashion-clip.pt"

CHECKPOINT_PATH = "../../models/checkpoints"
RESUME_CHECKPOINT = True

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Dino model configuration
CLOTHING_ITEMS_TO_DETECT = [
    ['a suit', 'a dress'],  # Cluster 1: sets
    ['a shirt', 'a blouse', 'a t-shirt', 'a sweater', 'a hoodie', 'a top'], # Cluster 2: tops
    ['a pair of pants', 'a skirt', 'a pair of shorts', 'a pair of trousers', 'a pair of jeans', 'a pair of leggings',], # Cluster 3: bottoms
    ['a coat', 'a jacket'], # Cluster 4: outerwear
    ['a pair of shoes'],  # Cluster 5: shoes
    ['a belt', 'a piece of jewelry', 'a bag', 'a hat', 'a scarf', 'a pair of sunglasses'], # Cluster 6: accessories
    ['a person', 'a full body wearing clothes']  # Cluster 7: full person
]

PERSON_TO_DETECT = 'a person wearing an outfit'

ITEMS_TO_DETECT = ['bottom wear', 'top wear', 'shirt', 't-shirt', 'pants', 'shoe', 'hat', 'jacket', 'dress', 'skirt', 'shorts', 'jeans', 'sweater', 'hoodie', 'coat', 'blouse', 'top', 'trousers', 'leggings', 'accessories', 'bag', 'backpack', 'belt', 'scarf', 'gloves', 'watch', 'jewelry', 'sunglasses']

METADATA_TAGS = ["casual", "bohemian", "classic", "streetwear", "preppy", "formal", "punk", "chic", "layered", "minimalist"]

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.7
CONFIDENCE_FILTER = 0.6
TARGET_SIZE = (224, 224)
EPOCHS = 100