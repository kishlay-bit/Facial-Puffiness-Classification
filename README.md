# Facial Puffiness Attribute Classification

Multi-label classification of three CelebA facial attributes — **Chubby**, **Double_Chin**, and **Bags_Under_Eyes** ("puffiness-adjacent" attributes) — using late fusion of a CNN image embedding with an 18-dim engineered geometric feature vector from MediaPipe Face Mesh landmarks. No clinical-diagnosis claim is made.

> Research conducted as part of an internship at SWAN Lab, Dept. of CSE, IIT Kharagpur.

## Key results

| Model | Type | Macro-F1 | mAP | Macro-AUC | Exact Match |
|---|---|---|---|---|---|
| **Ensemble-Scratch** | ensemble | **0.655** | 0.652 | 0.889 | 0.616 |
| **ResNet-Scratch** | scratch | 0.655 | 0.652 | **0.890** | **0.620** |
| Plain-CNN | scratch | 0.639 | 0.633 | 0.882 | 0.601 |
| Swin-Small | pretrained | 0.324 | 0.394 | 0.733 | 0.330 |
| EfficientNetV2-M | pretrained | 0.099 | 0.235 | 0.572 | 0.350 |

- From-scratch backbones (ResNet-Scratch, Plain-CNN) outperform pretrained ones on every metric under the current training budget — attributed to a convergence gap, not to pretraining being unhelpful in general.
- **f6** (lip-chin distance / inter-ocular distance) is the single most important geometric feature (0.098 macro-F1 drop on removal) — flagged in the report as needing reconciliation against a group-ablation number before being presented as headline.
- **Outer-eye-width** is a better normalization anchor than inter-ocular distance for these attributes (Section 5.2 of the report).
- Geometry's fusion value is inversely related to its standalone sufficiency: strongest fusion gain for Chubby, weakest (redundant with CNN texture) for Bags_Under_Eyes.
- Composite indices (FPI, CDI, UEPS) are reported as a **negative/null result**, not a contribution.

Full methodology, feature table, ablations, and discussion are in [`docs/technical_report.pdf`](docs/technical_report.pdf).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── docs/
│   └── technical_report.pdf              # full write-up: architecture, features, results, ablations, limitations
│
├── notebooks/
│   ├── 01_data_prep_landmarks.ipynb          # CelebA+MAAD-Face manifest, MediaPipe landmark/feature extraction
│   ├── 02_plain_cnn_scratch.ipynb            # from-scratch Plain-CNN fusion model
│   ├── 03_resnet_scratch.ipynb               # from-scratch ResNet fusion model (best single model)
│   ├── 04_swin_efficientnetv2_finetune.ipynb # pretrained backbones, fine-tuned
│   ├── 05_ensembling.ipynb                   # Ensemble-Scratch / Ensemble-All-4
│   ├── 06_feature_ablation.ipynb             # individual + group-level geometric feature ablation
│   └── 07_composite_indices_exploratory.ipynb # FPI / CDI / UEPS negative-result experiment
│
├── models/
│   ├── resnet_scratch_best.pt
│   ├── plain_cnn_best.pt
│   ├── swin_small_best.pt
│   ├── efficientnetv2_m_best.pt
│   └── README.md                         # checkpoint download links (see note below)
│
├── results/
│   ├── metrics_summary.csv               # main results table + per-attribute F1 breakdowns
│   ├── feature_ablation.csv              # individual + group feature-drop numbers
│   └── figures/
│       ├── fig1_overall_metrics.png
│       ├── fig2_per_attr_f1.png
│       ├── fig3_pr_curves.png
│       ├── fig4_roc_curves.png
│       ├── fig5_ablation_heatmap.png
│       ├── fig6_radar.png
│       ├── fig7_confusion_resnet_scratch.png
│       ├── fig8_scratch_vs_pretrained_gap.png
│       ├── figA_feature_importance_per_attr.png
│       ├── figB_importance_heatmap.png
│       ├── figC_top5_features.png
│       ├── figD_landmark_only_vs_cnn.png
│       ├── figE_novel_feature_distributions.png
│       └── figF_group_ablation.png
│
└── data/
    └── README.md                         # CelebA + MAAD-Face manifest source + landmark extraction steps
```

### Notes on what to actually commit

- **Don't commit model checkpoints or raw images directly** — host on Kaggle/Drive/HF Hub and link from `models/README.md` and `data/README.md`, or use Git LFS if you want them versioned in-repo.
- Several figures in your draft are still placeholder `[INSERT: ...]` tags rather than embedded images — export those plots from the notebooks before uploading so the filenames above actually exist.
- Rename exported PNGs (`image1.png`, `image14.png`, etc.) to the descriptive names above.

## Setup

```bash
git clone https://github.com/kishlay-bit/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Get the CelebA + MAAD-Face manifest and checkpoints (see `data/README.md` and `models/README.md`), then run notebooks in order (`01` → `07`).

## Method summary

- **Task:** multi-label binary classification of Chubby, Double_Chin, Bags_Under_Eyes on a combined ~150k-image CelebA + MAAD-Face manifest.
- **Architecture:** late-fusion — CNN backbone embedding concatenated with a 64-dim MLP output of 18 engineered MediaPipe geometric features, feeding three independent sigmoid heads (+ auxiliary gender head).
- **18 geometric features:** ratios/distances normalized by inter-ocular distance (IOD) or outer-eye width, grouped by targeted attribute (Chubby: f0–f3; Double_Chin: f4–f8; Bags_Under_Eyes: f9–f12; global reference: f13–f17). Landmark detection succeeded on 99.0% of images (197,781/199,829).
- **Backbones:** from-scratch Plain-CNN and ResNet-Scratch (13.7M params, residual connections); pretrained Swin-Small and EfficientNetV2-M, fine-tuned.
- **Training:** Asymmetric Loss, EMA, AdamW, CosineAnnealingWarmRestarts, WeightedRandomSampler.
- **Feature attribution:** ablation zeroes one feature/group at a time on the converged ResNet-Scratch model and measures macro-F1 drop from a 0.648 baseline.

## Limitations (see `docs/technical_report.pdf` §5.4, §6, §7 for full discussion)

- The f6 individual-feature drop (0.098) does not currently reconcile with the smaller group-level drop (0.034) for the group containing f6 — needs verification before being presented as a headline finding.
- Composite indices (FPI, CDI, UEPS) are a null result, not an improvement over the base 18 features.
- Pretrained backbones underperform substantially under the current fine-tuning budget — attributed to insufficient convergence, not used as evidence against pretraining generally.
- Several references (Face-to-BMI regression, MAAD-Face, periorbital puffiness clinical ref) are unverified placeholders pending completion.

## References

1. Liu, Z., Luo, P., Wang, X., Tang, X. (2015). Deep Learning Face Attributes in the Wild. *ICCV*.
2. Lingenfelter, B., Davis, S. R., Hand, E. M. (2022). A Quantitative Analysis of Labeling Issues in the CelebA Dataset. *ISVC*, LNCS vol. 13598.
3. Ridnik, T. et al. (2021). Asymmetric Loss for Multi-Label Classification. *ICCV*.
4. Liu, Z. et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV*.
5. Liu, Z. et al. (2022). A ConvNet for the 2020s (ConvNeXt). *CVPR*.
6. Tan, M., Le, Q. (2021). EfficientNetV2: Smaller Models and Faster Training. *ICML*.
7. Tan, M., Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.

## Author

**Kishlay Tejeswi** — B.Tech CSE, BIT Mesra · Research Intern, SWAN Lab, IIT Kharagpur
[GitHub](https://github.com/kishlay-bit)

## License

Add a license (MIT is a common default for research code) — see `LICENSE`.
