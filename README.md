# Iterative Outfit Completion (COS)

**What Happens When AI Tries Again?**  
*Moving Beyond Traditional Retrieval with Iterative Large-Scale CIR for Complete Outfit Suggestion*

This repository contains the **code implementation** accompanying the Master’s thesis:

> **Iterative, Compatibility-Guided Outfit Completion for Fashion Recommendation**  
> MSc in Engineering – Software Engineering for Intelligent Systems  
> University of Applied Sciences Vorarlberg  
> Author: Viktoriia Simakova

The project introduces a **human-inspired, iterative outfit refinement pipeline** that goes beyond single-step Complementary Item Retrieval (CIR) by repeatedly scoring, repairing and completing outfits until global coherence is achieved.

---

## Core Idea

Traditional fashion recommenders retrieve **one item once**. Humans don't style like that.

This project reframes outfit generation as a **structured decision-making problem**:

1. Start with a partial or flawed outfit
2. Score global compatibility
3. Identify the weakest element
4. Replace, add, or remove items
5. Re-evaluate and repeat

The process continues until the outfit is **complete, balanced, and coherent**.

---

## Key Features

- Iterative outfit refinement (add / remove / swap)
- Compatibility-guided decision loop
- FAISS-based large-scale retrieval
- Category-aware candidate filtering
- 8-Point Rule of Fashion constraint
- Batch compatibility scoring for efficiency
- Deterministic & reproducible experiments
- Quantitative + qualitative evaluation

---

## Models & Technologies

- **Backbone**: OutfitTransformer (set-wise transformer)
- **Embeddings**: FashionCLIP
- **Scoring**: Compatibility Prediction (CP)
- **Retrieval**: Complementary Item Retrieval (CIR)
- **Search**: FAISS
- **Dataset**: Polyvore (non-disjoint split)

---

## Installation

Follow the instructions for [OutfitTransformer](https://github.com/bigohofone/outfit-transformer). Then:

```bash
git clone https://github.com/s-tori2go/iterative-outfit-completion.git
```

---

## Results (Summary)

| Method | Mean Compatibility | Std Dev |
|------|-------------------|--------|
| Single-step CIR | 0.861 | 0.281 |
| Iterative COS | **0.992** | **0.066** |

- 86.2% of outfits improved
- Strong variance reduction
- Preferred by humans in 75% of cases

---

## Limitations

- Relies on Polyvore-era fashion data
- No explicit diversity regularization
- Metadata noise can cause category duplication

These are addressed in **Future Work**.

---

## Future Work

- Diversity-aware penalties
- User preference & context conditioning
- Modern outfit datasets
- Integration with VTON for full-outfit rendering
- Reinforcement learning for global optimization

---

## Citation

```bibtex
@mastersthesis{simakova2025iterative,
  title={What Happens When AI Tries Again? Iterative Large-Scale CIR for Complete Outfit Suggestion},
  author={Simakova, Viktoriia},
  year={2025},
  school={University of Applied Sciences Vorarlberg}
}
```

---

## License

MIT License

---

If you use or extend this work, please cite the thesis ✨

