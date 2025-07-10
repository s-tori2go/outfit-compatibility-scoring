# Personalized Outfit Evaluation with AI
_Advancing Fashion Styling with AI: Embedding-Aware Personal Style Modeling with Contextual Understanding_

## Overview

Welcome to the repository for my master's thesis on personalized fashion outfit evaluation using artificial intelligence! This project combines state-of-the-art deep learning models with user-centric design to assess and recommend outfits based on individual preferences and contextual cues.

## Project Structure

```
.
├── data/
│   └── [...]
├── models/
│   └── [...]
├── src/
│   └── [...]
├── environment.yml
├── README.md
└── ...
```

- **data/**: Contains all datasets, including images, metadata and extracted embeddings.
- **models/**: Stores loaded models and checkpoints for reproducibility and further experiments.
- **src/**: Core source code for data preparation, feature extraction, processing, scoring and visualization.

## Key Features

- **Multi-Modal Embedding Extraction**: Uses FashionCLIP to encode images and metadata into a shared vector space.
- **Personalized Style Anchors**: Learns user-specific "liked" and "disliked" style representations for tailored scoring.
- **Context-Aware Scoring**: Supports optional context prompts to adapt recommendations.
- **Interactive Visualization**: Provides 2D and 3D visualizations of the embedding space to explore style clusters and model behavior.
- **Baseline & Advanced Models**: Includes both a supervised ResNet50 baseline and a multi-modal, anchor-based scoring model.
- **Reproducible Environment**: All experiments run in a Conda environment, ensuring consistent results across systems.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/s-tori2go/outfit-compatibility-scoring.git
cd outfit-compatibility-scoring
```

### 2. Set Up the Environment

- Make sure you have [Conda](https://docs.conda.io/) installed.
- Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate outfit-style
```

- All dependencies (PyTorch, FashionCLIP, OpenCV, etc.) are specified in `environment.yml`.

### 3. Project Workflow

1. **Data Preparation**
   - Place raw images and metadata in `data/`.
   - Use scripts in `src/1a_2_data_preparation_module/` to crop, segment, and embed images and text.

2. **Model Training & Evaluation**
   - Train baseline and personalized models using scripts in `src/3_4_scoring_module/`.
   - Evaluate and visualize results with provided notebooks and scripts.

3. **Scoring New Outfits**
   - Use the scoring pipeline to rate new outfits based on your personalized style anchors and (optionally) context.

## Results & Insights

- Achieves ~75% accuracy on unseen outfit data, reflecting the subjective nature of style.
- Visualizations reveal partial clustering of "good" and "bad" outfits, with clear separation of personalized anchors.
- Both baseline and advanced models are included for benchmarking and comparison.

## Why This Matters

This project lays the groundwork for AI-powered, user-centric fashion assistants. These are tools that not only recognize general style but also adapt to what _you_ like, in any context. The modular pipeline and transparent code make it easy to extend, retrain, or adapt for new users and datasets.

## Acknowledgements

- Developed as part of the Master’s thesis at University of Applied Sciences Vorarlberg.
- Special thanks to Dr. techn. Sebastian Hegenbart for supervision.

## Keywords

_Fashion AI, Deep Learning, Personalization, Outfit Compatibility, Multi-Modal Embeddings, FashionCLIP, Computer Vision, Context-Aware Scoring_

For any questions or contributions, feel free to open an issue or contact the author!