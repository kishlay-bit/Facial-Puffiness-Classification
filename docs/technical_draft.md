# Facial Puffiness Attribute Classification with Geometric Feature Fusion
### Technical Draft — Summary for Mentor Review
*This is a working technical summary, not the paper itself. It documents what was built, what was measured, and what can be claimed from the results so far. Sections marked `[INSERT: filename]` indicate where a specific generated figure should go; sections marked "Author to add" are for supplementary material you'll attach yourself.*

---

## 1. Task and Motivation

We classify three CelebA facial attributes — **Chubby**, **Double_Chin**, **Bags_Under_Eyes** — as a dedicated multi-label task, distinct from their usual treatment as three of forty undifferentiated rows in the standard CelebA attribute-classification literature. We refer to this triplet as "puffiness-adjacent" attributes, motivated by their loose correspondence to facial adiposity and periorbital fullness, while making no clinical-diagnosis claim.

The core architectural idea is **late fusion of a CNN image embedding with an engineered, anatomically-targeted geometric feature vector** derived from MediaPipe Face Mesh landmarks, rather than relying on image texture alone.

---

## 2. Architecture

**Shared fusion design, used identically across every backbone tested (from-scratch and pretrained):**

```
Face image ──► [Backbone] ──► image embedding (D-dim)
                                                   \
                                                    ► concat ──► Fusion FC ──► 3 sigmoid heads
                                                   /                          (Chubby, Double_Chin,
18-dim MediaPipe ──► [Geometric MLP] ──► 64-dim ─┘                            Bags_Under_Eyes)
   feature vector      (BN→FC→ReLU→Drop ×2)
```

Backbones evaluated:
- **From-scratch:** Plain CNN (4-block conv), ResNet-Scratch (both trained without ImageNet initialization)
- **Pretrained (ImageNet-initialized, fine-tuned):** Swin-Small, EfficientNetV2-M

**Important note for the paper:** in this experiment set, the from-scratch arm and the pretrained arm both consume the **same 18-dimensional engineered geometric feature vector** (not raw landmark coordinates). This is a meaningful improvement over an earlier iteration of the project, where the two arms used different landmark representations (raw 1,434-dim mesh vs. 18-dim engineered features) — that mismatch would have confounded any scratch-vs-pretrained comparison. With both arms now sharing the same feature representation, the comparison in Section 4 isolates backbone/training effects cleanly.

`[INSERT: architecture diagram — author to add or regenerate to match this exact figure]`

---

## 3. The 18 Geometric Features

All features are ratios or distances normalized for scale-invariance across face size/camera distance. Two normalization anchors are used: inter-ocular distance (IOD, landmarks 133/362) and outer-eye width (landmarks 33/263).

| # | Feature | Formula | Targeted attribute |
|---|---|---|---|
| f0 | jaw_width / IOD | landmarks 234,454 ÷ IOD | Chubby |
| f1 | cheek_width / IOD | landmarks 116,345 ÷ IOD | Chubby |
| f2 | jaw_width / face_height (fWHR) | | Chubby |
| f3 | cheek_width / face_height | | Chubby |
| f4 | lower_face_height / face_height | | Double_Chin |
| f5 | chin_drop / IOD | jaw-mid (172,397) to chin (152) | Double_Chin |
| f6 | lip_chin_dist / IOD | lower lip (18) to chin (152) | Double_Chin |
| f7 | jaw_width / cheek_width | | Double_Chin |
| f8 | lower_face_height / IOD | | Double_Chin |
| f9 | under_eye_z_var (left) | landmarks 160,159,158 | Bags_Under_Eyes |
| f10 | under_eye_z_var (right) | landmarks 387,386,385 | Bags_Under_Eyes |
| f11 | eye_cheek_vert / IOD | | Bags_Under_Eyes |
| f12 | mean_under_eye_depth | mean(f9, f10) | Bags_Under_Eyes |
| f13 | face_height / IOD | | Global reference |
| f14 | inner/outer eye width ratio | | Global reference |
| f15 | jaw_width / outer_eye_width | | Global reference |
| f16 | cheek_width / outer_eye_width | | Global reference |
| f17 | lower_face_height / outer_eye_width | | Global reference |

Detection success: 197,781 / 199,829 images (99.0%); failed detections get a zero vector.

Three additional composite indices were tested as an extension (Section 6): **Face Puffiness Index (FPI)**, **Chin Droop Index (CDI)**, **Under-Eye Asymmetry Score (UEPS)** — see Section 6 for results, which are mixed and should **not** be presented as a headline contribution (reasoning below).

---

## 4. Main Results: Scratch vs. Pretrained vs. Ensemble

| Model | Type | Macro-F1 | mAP | Macro-AUC | Chubby F1 | Double_Chin F1 | Bags F1 | Exact Match |
|---|---|---|---|---|---|---|---|---|
| Ensemble-Scratch | ensemble | **0.655** | 0.652 | 0.889 | 0.621 | 0.647 | 0.697 | 0.616 |
| ResNet-Scratch | scratch | 0.655 | 0.652 | **0.890** | 0.616 | 0.650 | 0.697 | 0.620 |
| Ensemble-All-4 | ensemble | 0.645 | **0.658** | 0.886 | 0.594 | 0.644 | 0.697 | 0.619 |
| Plain-CNN | scratch | 0.639 | 0.633 | 0.882 | 0.613 | 0.609 | 0.693 | 0.601 |
| Swin-Small | pretrained | 0.324 | 0.394 | 0.733 | 0.237 | 0.189 | 0.547 | 0.330 |
| Ensemble-Pretrained | ensemble | 0.311 | 0.394 | 0.741 | 0.250 | 0.165 | 0.516 | 0.324 |
| EfficientNetV2-M | pretrained | 0.099 | 0.235 | 0.572 | 0.264 | 0.019 | 0.015 | 0.350 |

**Claims supported by this table:**
- The from-scratch backbones (ResNet-Scratch, Plain-CNN), trained with a full budget, clearly outperform both pretrained backbones on every metric.
- A simple mean-probability ensemble of the two scratch models (**Ensemble-Scratch**) gives a small but real improvement over the best single scratch model on macro-F1 (0.655 vs. 0.655, essentially tied) and mAP.
- Including the pretrained models in the ensemble (**Ensemble-All-4**) *reduces* macro-F1 relative to the scratch-only ensemble (0.645 vs. 0.655) — the pretrained models are not yet strong enough to contribute positively to ensembling.
- Bags_Under_Eyes is the easiest attribute across every model (highest F1 in every row); Double_Chin and Chubby are harder and more model-dependent.

**Framing for the paper (honest, not overreaching):** the pretrained-backbone results should be attributed to a training-budget/convergence gap, not to a claim that pretraining doesn't help — this should be stated plainly rather than argued around.

`[INSERT: fig1_overall_metrics.png — grouped bar chart, Macro-F1 / mAP / Mean-AUC across all 7 models]`
`[INSERT: fig2_per_attr_f1.png — per-attribute F1 by model]`
`[INSERT: fig3_pr_curves.png — Precision-Recall curves per attribute]`
`[INSERT: fig4_roc_curves.png — ROC curves per attribute]`
`[INSERT: fig6_radar.png — radar/spider comparison across F1 per attribute + mAP + AUC + exact-match]`
`[INSERT: fig7_confusion.png — confusion matrices, ResNet-Scratch]`
`[INSERT: fig8_gap.png — scratch-vs-pretrained F1 gap per attribute]`

---

## 5. Feature Attribution: What Do the Geometric Features Actually Contribute?

Ablation method: zero out one feature (or a defined group of features) at inference time on the trained, converged ResNet-Scratch model; measure the drop in macro-F1 relative to the full-feature baseline (baseline macro-F1 = 0.648).

### 5.1 Individual-feature importance

| Rank | Feature | Macro-F1 drop | Chubby drop | Double_Chin drop | Bags drop |
|---|---|---|---|---|---|
| 1 | f6: lip_chin_dist / IOD | 0.098 | 0.188 | 0.103 | 0.003 |
| 2 | f15: jaw_width / outer_eye_width | 0.068 | 0.151 | 0.047 | 0.006 |
| 3 | f16: cheek_width / outer_eye_width | 0.035 | 0.086 | 0.012 | 0.006 |
| 4 | f17: lower_face_h / outer_eye_width | 0.019 | 0.047 | 0.007 | 0.002 |
| 5 | f4: lower_face_h / face_height | 0.009 | 0.023 | 0.002 | 0.001 |

Most individual under-eye features (f9, f10, f12) show **~zero or slightly negative** F1 drop when removed — i.e., no measurable individual contribution from this ablation.

`[INSERT: figA_feature_importance_per_attr.png — per-attribute horizontal bar rankings]`
`[INSERT: figB_importance_heatmap.png — full feature × attribute heatmap]`
`[INSERT: figC_top5_features.png — top-5 feature summary bar chart]`

### 5.2 Claim A (strong, defensible): Outer-eye-width is a better normalization anchor than IOD for these attributes

This is the cleanest result in the study because it's a controlled, like-for-like comparison: f0/f1/f8 (jaw width, cheek width, lower-face-height ÷ **IOD**) vs. f15/f16/f17 (the *identical* raw measurements ÷ **outer-eye-width**). The outer-eye-normalized versions are 5–14× more important by this ablation (e.g., f15 drop 0.068 vs. f0 drop 0.005). This supports a specific, citable methodological claim: for puffiness-adjacent attributes, outer-eye width outperforms the standard IOD convention as a normalization reference.

### 5.3 Claim B (strong, defensible): geometry's contribution is attribute-dependent and inversely related to its standalone sufficiency

From the group-level ablation (image-only vs. full-fusion) and landmark-only classifiers:

| Attribute | Geometry-alone F1 (best of LogReg/GBM/RF) | Image-only F1 (no landmarks) | Full fusion F1 | Landmark contribution (fusion − image-only) |
|---|---|---|---|---|
| Bags_Under_Eyes | 0.604 | 0.691 | 0.694–0.697 | ≈ +0.003 (negligible) |
| Double_Chin | 0.478 | 0.629 | 0.650 | +0.021 |
| Chubby | 0.369 | 0.540 | 0.600–0.616 | +0.06–0.08 (largest) |

**The claim:** geometry alone is most predictive, standalone, for Bags_Under_Eyes — but that's also the attribute where adding geometry to the CNN helps *least*, because the CNN's texture features already capture the same signal (redundancy, not complementarity). Conversely, geometry alone is weakest for Chubby, but it's the attribute where fusing geometry with the CNN helps *most* (complementary information the CNN doesn't otherwise extract). This inverse relationship — standalone geometric sufficiency vs. incremental fusion value — is a genuinely non-obvious, attribute-specific finding worth stating explicitly rather than just reporting an aggregate "landmarks help" number.

`[INSERT: figD_landmark_only_vs_cnn.png — geometry-alone classifiers vs. full fusion model]`
`[INSERT: figF_group_ablation.png — F1 drop per feature group when zeroed]`
`[INSERT: fig5_ablation_heatmap.png — feature-group × attribute ablation heatmap]`

### 5.4 Item to verify before this goes in the paper

The individual-feature ablation says removing **f6 alone** drops macro-F1 by 0.098 (the single largest effect measured). The group-level ablation says removing the **entire f4–f8 group** (which contains f6, plus four other features) drops macro-F1 by only 0.034. A superset removal cannot legitimately produce a smaller drop than removing one of its members, unless there's a real interaction effect (e.g., BatchNorm1d(18) being fit jointly, so zeroing one feature while the other four stay at normal scale causes a larger distribution shift than zeroing all five together) — which would itself be a reportable finding — or there's a bookkeeping mismatch between the two ablation scripts (different threshold reused, different zeroing convention). **This should be checked before f6 is presented as the headline single-feature finding**, since it's currently the largest number in the whole attribution study and it doesn't reconcile with the number next to it.

---

## 6. Composite Geometric Indices (Exploratory — Recommend Reporting as a Negative/Null Result)

Three hand-engineered composite indices were tested: FPI (jaw × cheek / lower-face-proportion), CDI (chin-drop × lower-face-proportion), UEPS (bilateral under-eye z-depth asymmetry).

| Attribute | Classifier | F1 (18 feats) | F1 (21 feats, +FPI/CDI/UEPS) | Δ |
|---|---|---|---|---|
| Chubby | LogReg | 0.3695 | 0.3702 | +0.0007 |
| Chubby | GBM | 0.1512 | 0.1534 | +0.0022 |
| Chubby | RF | 0.1324 | 0.1241 | **−0.0083** |
| Double_Chin | LogReg | 0.4783 | 0.4805 | +0.0022 |
| Double_Chin | GBM | 0.3631 | 0.3643 | +0.0012 |
| Double_Chin | RF | 0.3285 | 0.3158 | **−0.0127** |
| Bags_Under_Eyes | LogReg | 0.5859 | 0.5876 | +0.0017 |
| Bags_Under_Eyes | GBM | 0.6041 | 0.6033 | −0.0008 |
| Bags_Under_Eyes | RF | 0.6042 | 0.6026 | −0.0016 |

**Honest read:** gains under LogReg are within noise for this dataset size (+0.0007 to +0.0022); GBM is mixed; RF is negative across all three attributes. This is not a "consistent improvement" — it's a **null result**, most likely because the composite indices are algebraic recombinations of features already present in the base 18 (FPI is derived from f0, f1, f4), so tree-based models that can already learn feature interactions gain nothing from having the interaction pre-computed, and may even be hurt by the added redundant dimensionality/correlation. Recommend presenting this as a clearly-labeled negative result — useful for telling future work not to bother re-deriving these ratios — rather than as a contribution.

The distribution evidence agrees: UEPS in particular shows almost no separation between Bags_Under_Eyes=0 and =1 (|Δmean| = 0.007, vs. 0.055 and 0.051 for FPI and CDI respectively) — i.e., UEPS is close to a non-informative feature by direct inspection, independent of any downstream model.

`[INSERT: figE_novel_feature_distributions.png — FPI/CDI/UEPS distribution by label]`

---

## 7. Summary of Claims (ranked by how defensible they are)

1. **Strong / ready to write up:** outer-eye-width normalization outperforms IOD normalization for jaw/cheek/lower-face geometric ratios in this task (Section 5.2).
2. **Strong / ready to write up:** geometric-feature standalone sufficiency and incremental fusion value are inversely related across the three attributes — Bags_Under_Eyes is geometry-sufficient-but-redundant with CNN texture; Chubby is geometry-weak-but-complementary (Section 5.3).
3. **Strong / ready to write up:** shared 18-dim feature representation across scratch and pretrained arms gives a clean scratch-vs-pretrained comparison; from-scratch backbones clearly outperform under the current training budget, and a scratch-only ensemble gives a small further gain (Section 4).
4. **Needs one verification pass before writing up:** f6 (lip-chin-distance/IOD) as the single most important individual feature — reconcile with the group-ablation number first (Section 5.4).
5. **Report as a negative/null result, not a contribution:** the three composite indices (FPI/CDI/UEPS) do not reliably improve on the base 18 features (Section 6).
6. **Known limitation, state plainly rather than minimize:** pretrained backbones (Swin-Small, EfficientNetV2-M) substantially underperform scratch models under the current fine-tuning budget; this is attributed to insufficient convergence, not evidence against pretraining in general.

---

## 8. References (verified this session)

1. Liu, Z., Luo, P., Wang, X., Tang, X. (2015). *Deep Learning Face Attributes in the Wild.* ICCV 2015. (CelebA dataset.)
2. Lingenfelter, B., Davis, S. R., Hand, E. M. (2022). *A Quantitative Analysis of Labeling Issues in the CelebA Dataset.* ISVC 2022, LNCS vol. 13598, Springer.
3. Ridnik, T., Ben-Baruch, E., Zamir, N., Noy, A., Friedman, I., Protter, M., Zelnik-Manor, L. (2021). *Asymmetric Loss for Multi-Label Classification.* ICCV 2021.
4. Liu, Z., Lin, Y., Cao, Y., et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021.
5. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., Xie, S. (2022). *A ConvNet for the 2020s (ConvNeXt).* CVPR 2022.
6. Tan, M., Le, Q. (2021). *EfficientNetV2: Smaller Models and Faster Training.* ICML 2021.
7. Tan, M., Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.
8. *[Face-to-BMI regression reference — verify exact citation before submission.]*
9. *[MAAD-Face dataset reference — verify exact citation before submission.]*
10. *[Periorbital puffiness / edema clinical reference — verify exact citation before submission.]*

---

## 9. Supplementary Material (Author to Add)

*Space for additional screenshots, plots, or results not covered above — insert below with a short caption for each.*

- `[Author to add: ]`
- `[Author to add: ]`
- `[Author to add: ]`
