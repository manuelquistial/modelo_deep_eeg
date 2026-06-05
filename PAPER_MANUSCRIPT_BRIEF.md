# Experimental Reference — Motor Imagery Classification Benchmark

Complete factual record of the study: datasets, preprocessing, models, training, evaluation protocol, and hold-out results. Intended as input material for manuscript preparation.

**Evaluation scope:** subject-independent 80/20 hold-out; trial-level classification; two datasets (PhysioNet MI, BNCI2014-001); four models; Euclidean Alignment ablation (on/off).

**Literature scope:** positioning against published MI benchmarks (12 papers reviewed from Zotero library; PDFs and extraction in `literature_comparison/`). Literature numbers are cited for context only when protocol alignment is stated explicitly.

---

## 1. Study Design

| Item | Value |
|------|--------|
| Paradigm | Motor imagery (MI), binary classification |
| Classes | Left hand vs. right hand |
| Prediction granularity | One label per EEG trial (epoch) |
| Input shape | `(N, C, T)` → deep models receive `(batch, 1, C, T)` |
| Split type | Subject-disjoint hold-out: 80% DEV / 20% TEST (`test_size=0.20`, `random_state=42`) |
| Deep learning validation | 15% of DEV subjects held for validation (`val_ratio=0.15`, stratified by class) |
| Classical baselines training | All DEV trials (train + validation subjects combined) |
| Test evaluation | All trials from TEST subjects only |
| Fair comparison | All models consume identical preprocessed cached arrays per dataset × EA condition |
| Ablation factor | Euclidean Alignment: `use_ea ∈ {true, false}` |
| Models compared | EEGMeModel, EEGNet, FBCSP+LDA, CSP+SVM |
| Metrics | Accuracy, balanced accuracy, macro F1, Cohen's κ |

---

## 2. Datasets

### 2.1 PhysioNet EEG Motor Movement/Imagery Dataset

| Property | Value |
|----------|--------|
| Access | MOABB `PhysionetMI`, MNE EEGBCI downloader |
| Original sampling rate | 160 Hz |
| Subjects included | 108 |
| Subjects excluded | Subject 88 (recorded at 128 Hz; incompatible with 160 Hz cohort) |
| EEG channels | 64 |
| Motor imagery classes retained | `left_hand`, `right_hand` |
| Excluded event types | Feet, tongue, rest, open/close fist (binary L/R only) |
| Cached DEV set | 3,859 trials, 86 subjects |
| Cached TEST set | 990 trials, 22 subjects |
| Time samples per trial (after preprocessing) | 380 (center-cropped from 385; aligned to multiple of 10 for CNN pooling) |
| `common_n_times_raw` | 385 |

**DEV subject IDs (86):**  
1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 16, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 35, 37, 40, 41, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 90, 91, 92, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108

**TEST subject IDs (22):**  
8, 10, 14, 15, 18, 21, 30, 34, 36, 38, 39, 42, 47, 60, 69, 86, 87, 89, 93, 94, 99, 109

**Channel montage (64):**  
FC5, FC3, FC1, FCz, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, C6, CP5, CP3, CP1, CPz, CP2, CP4, CP6, Fp1, Fpz, Fp2, AF7, AF3, AFz, AF4, AF8, F7, F5, F3, F1, Fz, F2, F4, F6, F8, FT7, FT8, T7, T8, T9, T10, TP7, TP8, P7, P5, P3, P1, Pz, P2, P4, P6, P8, PO7, PO3, POz, PO4, PO8, O1, Oz, O2, Iz

### 2.2 BNCI2014-001 Dataset

| Property | Value |
|----------|--------|
| Access | MOABB `BNCI2014_001` |
| Subjects | 9 (full dataset) |
| EEG channels | 22 |
| Paradigm time window | 0.0 s to 4.0 s post-cue (`bnci_tmin`, `bnci_tmax`) |
| Resampling (MOABB paradigm) | 125 Hz (`bnci_resample`) |
| Paradigm band-pass | 1–40 Hz (`bnci_fmin`, `bnci_fmax`) |
| Classes | Left hand, right hand (`bnci_n_classes=2`) |
| Cached TEST trials (hold-out) | 576 (288 left_hand, 288 right_hand) |

**Channel montage (22):**  
Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz

---

## 3. Label Encoding

| Representation | Left hand | Right hand |
|----------------|-----------|------------|
| MOABB event name | `left_hand` | `right_hand` |
| PyTorch class index (deep models) | 0 | 1 |
| Workshop / CSP label (baselines) | 1 | 2 |

---

## 4. Preprocessing Pipeline

Applied identically before caching. All models read the same cached tensors for a given dataset and EA condition.

### 4.1 Sequence

1. **Outlier rejection** — discard trials with peak amplitude > 800 µV; automatic detection of Volts vs. microvolts scaling.
2. **High-pass filter** — 4 Hz Butterworth, order 4, zero-phase (`scipy.signal.sosfiltfilt`).
3. **Time harmonization** — crop all trials to cohort minimum length; center crop when longer (`crop_mode: crop`).
4. **CNN length alignment** — crop to largest multiple of 10 samples (required by `AvgPool2d(kernel=(1,10))` in EEGMeModel).
5. **Euclidean Alignment (optional)** — per-subject EA on DEV and TEST separately; reference covariance from subject trials; regularization `ea_reg = 1e-10`; alignment verified (`ea_verify_enabled: true`).
6. **Hold-out subject split** — stratified by majority class per subject (`sklearn.model_selection.train_test_split`).

### 4.2 Configuration files

| Condition | YAML config | `use_ea` |
|-----------|-------------|----------|
| With EA | `configs/preprocess_ea.yaml` | `true` |
| Without EA | `configs/preprocess_no_ea.yaml` | `false` |
| BNCI dataset selector | `configs/dataset_bnci2014_001.yaml` | inherits from preprocess config |
| EEGNet model selector | `configs/model_eegnet.yaml` | — |

### 4.3 Cache artifacts

| Dataset | EA | Cache directory |
|---------|-----|---------------|
| PhysioNet | yes | `data/processed/physionet_lr_ea_108sub_all/` |
| PhysioNet | no | `data/processed/physionet_lr_no_ea_108sub_all/` |

Each cache contains `holdout/X_dev.npy`, `y_dev.npy`, `groups_dev.npy`, `X_test.npy`, `y_test.npy`, `groups_test.npy`, `meta.json`.

---

## 5. Evaluation Protocol

### 5.1 Data flow

```
Cohort (all subjects)
│
├── TEST subjects (20%) ──────────────────► Final evaluation (all TEST trials)
│
└── DEV subjects (80%)
    ├── TRAIN subjects (~85% of DEV) ─────► Deep model training
    └── VAL subjects (~15% of DEV) ───────► Early stopping / LR scheduling
```

- No subject overlap between TRAIN, VAL, and TEST.
- Metrics computed over all TEST trials (trial-level aggregation).
- Classical baselines: trained on full DEV (TRAIN + VAL subjects), evaluated on TEST.

### 5.2 Validation split (deep learning only)

- Function: `split_val_subjects_stratified`
- `val_ratio = 0.15`
- `random_state = 42` (same as hold-out; independent stratification within DEV)
- Guarantees ≥2 validation subjects when both classes are present

---

## 6. Model Specifications

### 6.1 EEGMeModel (CNN + Transformer)

| Component | Specification |
|-----------|---------------|
| Origin | Ported from project `funtions.py` / `code.ipynb` |
| LocalFeatureLearner | Temporal `Conv2d(1→F1, k=(1,10), pad=(0,5))` → BN → ELU → Spatial `Conv2d(F1→F1, k=(C,1))` → BN → ELU → `AvgPool2d(1,10)` → Dropout |
| Projection | `Conv2d(F1→embed_dim, k=(1,1))` |
| Transformer | 2× `TransformerEncoderLayer`, `d_model=embed_dim`, `nhead=4`, GELU, `batch_first=True` |
| Head | Temporal global average → L2-normalized embedding → `Linear(embed_dim, 2)` |
| Output | `(logits, embeddings, local_features, global_features)` |
| Constraint | `n_times` must be divisible by 10 |

| Hyperparameter | PhysioNet | BNCI |
|----------------|-----------|------|
| `F1` | 7 | 7 |
| `embed_dim` | 32 | 32 |
| `dropout` | 0.25 | 0.25 |
| `n_channels` | 64 | 22 |
| `n_times` | 380 | paradigm-dependent (post-cache) |
| `num_classes` | 2 | 2 |

### 6.2 EEGNet (Lawhern et al., 2018)

| Component | Specification |
|-----------|---------------|
| Block 1 | `Conv2d(1→F1, (1, kernel_length))` → BN → Depthwise `Conv2d(F1→F1·D, (C,1), groups=F1)` → BN → ELU → `AvgPool2d(1,4)` → Dropout |
| Block 2 | Separable `Conv2d(F1·D→F2, (1,16), pad=(0,8))` → BN → ELU → `AvgPool2d(1,8)` → Dropout |
| Head | Flatten → `Linear` → 2 classes |
| Output | `(logits, embeddings)` |

| Hyperparameter | Value |
|----------------|--------|
| `F1` | 8 |
| `D` | 2 |
| `F2` | 16 |
| `kernel_length` | 64 |
| `dropout` | 0.5 |

### 6.3 FBCSP + LDA

| Component | Specification |
|-----------|---------------|
| Filter bank | 8 bands (Hz): [4,8], [8,12], [12,16], [16,20], [20,24], [24,28], [28,32], [32,36] |
| Band-pass | Butterworth, order 5, per band |
| CSP per band | 4 components, `reg='ledoit_wolf'`, `log=True`, `norm_trace=True`, `transform_into='average_power'` |
| Total CSP features | 32 |
| Feature selection | `SelectKBest` with mutual information, k=16 |
| Scaling | `StandardScaler` (fit DEV, apply TEST) |
| Classifier | `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])` |

### 6.4 CSP + SVM

| Component | Specification |
|-----------|---------------|
| CSP | 4 components, `log=True`, `norm_trace=False`, `reg=None` |
| Pipeline | CSP → StandardScaler → SVC(RBF) |
| Hyperparameter search | `GridSearchCV`, 3-fold stratified CV on DEV |
| SVM `C` grid | {0.1, 1, 10} |
| SVM `gamma` grid | {scale, auto, 0.01, 0.1} |
| Scoring | Accuracy |
| `n_jobs` | -1 |

---

## 7. Deep Learning Training

| Parameter | Value |
|-----------|--------|
| Optimizer | Adam (`lr=0.001`) |
| Batch size | 32 |
| Max epochs | 100 |
| Loss | `CrossEntropyLoss` |
| Early stopping | Patience 20 on validation loss |
| LR scheduler | `ReduceLROnPlateau` (patience 10, factor 0.1, mode=min) |
| Trial normalization | Per-trial z-score per channel (`trial_wise_normalize: true`) |
| Checkpoint selection | Lowest validation loss |
| Random seed | 42 (`seed_everything`: Python, NumPy, PyTorch, CUDA) |
| Device | Auto-detect: CUDA → MPS → CPU |
| Compute environment | Paperspace Gradient GPU; PyTorch built with CUDA 12.4 wheels |

---

## 8. Software and Reproducibility

| Item | Detail |
|------|--------|
| Repository | https://github.com/manuelquistial/modelo_deep_eeg |
| Python | ≥ 3.11 |
| Core packages | `mne≥1.6`, `moabb≥1.0`, `scikit-learn≥1.4`, `numpy≥1.26,<2`, `scipy≥1.11`, `pandas≥2.0`, `pytorch≥2.0` |
| Data preparation | `python scripts/prepare_data.py --config <yaml>` |
| Deep training | `python scripts/train_holdout.py --config <yaml>` |
| LDA baseline | `python scripts/run_baseline_lda.py --config <yaml>` |
| CSP+SVM baseline | `python scripts/run_baseline_csp_svm.py --config <yaml>` |
| Full comparison table | `outputs/pipeline_comparison.csv` |

---

## 9. Hold-Out Results

**Source:** `outputs/pipeline_comparison.csv`  
**Protocol:** hold-out  
**Aggregation:** trial-level metrics on TEST subjects

### 9.1 PhysioNet MI — full results

| Model | EA | Accuracy | Balanced Acc. | Macro F1 | κ | Best epoch |
|-------|:--:|---------:|--------------:|---------:|--:|-----------:|
| CSP+SVM | yes | 0.7091 | 0.7089 | 0.7089 | 0.418 | — |
| FBCSP+LDA | yes | 0.7030 | 0.7023 | 0.7019 | 0.405 | — |
| EEGNet | yes | 0.6818 | 0.6823 | 0.6816 | 0.364 | 20 |
| EEGMeModel | yes | 0.6465 | 0.6481 | 0.6427 | 0.295 | 8 |
| CSP+SVM | no | 0.6374 | 0.6374 | 0.6373 | 0.275 | — |
| EEGNet | no | 0.6212 | 0.6215 | 0.6212 | 0.243 | 23 |
| EEGMeModel | no | 0.6212 | 0.6200 | 0.6178 | 0.241 | 33 |
| FBCSP+LDA | no | 0.5869 | 0.5883 | 0.5832 | 0.176 | — |

**PhysioNet — accuracy change with EA (percentage points):**

| Model | No EA | With EA | Δ |
|-------|------:|--------:|--:|
| FBCSP+LDA | 58.69% | 70.30% | +11.61 |
| CSP+SVM | 63.74% | 70.91% | +7.17 |
| EEGNet | 62.12% | 68.18% | +6.06 |
| EEGMeModel | 62.12% | 64.65% | +2.53 |

**Independent replication (PhysioNet, EEGMeModel + EA, GPU):** accuracy = 0.6576, κ = 0.315, best epoch = 15.

### 9.2 BNCI2014-001 — full results

| Model | EA | Accuracy | Balanced Acc. | Macro F1 | κ | Best epoch |
|-------|:--:|---------:|--------------:|---------:|--:|-----------:|
| EEGNet | yes | 0.8403 | 0.8403 | 0.8403 | 0.681 | 66 |
| EEGNet | no | 0.8160 | 0.8160 | 0.8160 | 0.632 | 76 |
| EEGMeModel | yes | 0.7135 | 0.7135 | 0.7133 | 0.427 | 43 |
| EEGMeModel | no | 0.6892 | 0.6892 | 0.6878 | 0.378 | 98 |
| CSP+SVM | yes | 0.6337 | 0.6337 | 0.6316 | 0.267 | — |
| FBCSP+LDA | yes | 0.6233 | 0.6233 | 0.6171 | 0.247 | — |
| FBCSP+LDA | no | 0.5868 | 0.5868 | 0.5857 | 0.174 | — |
| CSP+SVM | no | 0.5833 | 0.5833 | 0.5494 | 0.167 | — |

**BNCI — accuracy change with EA (percentage points):**

| Model | No EA | With EA | Δ |
|-------|------:|--------:|--:|
| EEGNet | 81.60% | 84.03% | +2.43 |
| EEGMeModel | 68.92% | 71.35% | +2.43 |
| FBCSP+LDA | 58.68% | 62.33% | +3.65 |
| CSP+SVM | 58.33% | 63.37% | +5.04 |

### 9.3 Model ranking by accuracy (with EA)

| Rank | PhysioNet | BNCI2014-001 |
|------|-----------|--------------|
| 1 | CSP+SVM (70.91%) | EEGNet (84.03%) |
| 2 | FBCSP+LDA (70.30%) | EEGMeModel (71.35%) |
| 3 | EEGNet (68.18%) | CSP+SVM (63.37%) |
| 4 | EEGMeModel (64.65%) | FBCSP+LDA (62.33%) |

### 9.4 EEGNet vs. EEGMeModel gap (with EA)

| Dataset | EEGNet | EEGMeModel | Δ (pp) |
|---------|-------:|-----------:|-------:|
| PhysioNet | 68.18% | 64.65% | +3.53 |
| BNCI | 84.03% | 71.35% | +12.68 |

---

## 10. Literature Context and Study Positioning

### 10.1 Why literature context is required

Published MI classification accuracies are not interpretable in isolation. The same dataset can yield 60–97% depending on evaluation protocol (within-subject CV vs. cross-subject hold-out vs. LOSO), number of classes (2 vs. 4), preprocessing (with/without EA), and whether train and test subjects are disjoint. This study is designed as a **controlled benchmark**: four representative pipelines (two classical, two deep) evaluated under **identical cached preprocessing** on two public datasets with an **EA ablation**, using a **fixed subject-disjoint 80/20 hold-out**.

### 10.2 State of the art (concise map)

| Line of work | Representative references | Typical evaluation | Relevance to this study |
|--------------|---------------------------|--------------------|-------------------------|
| Classical spatial filtering | Ramoser et al. 2000; Blankertz et al. 2008; Ang et al. 2008 (FBCSP) | Often within-subject or competition splits | Baselines FBCSP+LDA and CSP+SVM implemented here |
| Deep learning for EEG/MI | Schirrmeister et al. 2017; Lawhern et al. 2018 (EEGNet) | Mixed; frequently subject-dependent | EEGNet included as standard deep baseline |
| Cross-subject transfer / alignment | He & Wu 2020 (EA); She et al. 2023 (domain adaptation) | LOSO or session transfer | EA ablation directly tests alignment effect |
| Recent deep MI decoders | Altaheri et al. 2023; Singh et al. 2025; Zhang et al. 2026; Pan et al. 2026 | Often BCI Competition IV-2a, 4-class, LOSO or SD | Context for BNCI results; protocol usually differs |
| PhysioNet MI benchmarks | Kim et al. 2016; Das et al. 2025; Pan et al. 2026 | Variable splits; often within-subject or unclear subject_disjoint | Context for PhysioNet results |
| Reproducible BCI tooling | Jayaram et al. 2018 (MOABB) | Pipeline standardization | Data access and paradigm definition |
| Systematic reviews | Saibene et al. 2024; Hameed et al. 2025 | Aggregated heterogeneous protocols | Frame expected accuracy ranges |

### 10.3 Gap addressed by this benchmark

Most prior MI studies report a **single model** on **one dataset** with a **dataset-specific protocol**. Few studies simultaneously satisfy all of the following:

1. **Subject-disjoint evaluation** (no trial leakage across subjects in test set).
2. **Identical preprocessing** consumed by classical and deep models.
3. **Direct EA on/off comparison** on the same cached tensors.
4. **Cross-dataset comparison** (large cohort PhysioNet vs. small BNCI2014-001).
5. **Binary L/R MI** (clinically common paradigm; many papers use 4-class BCI-2a).

This benchmark does not claim state-of-the-art accuracy on either dataset. Its contribution is **comparative, reproducible evidence** of which model family performs best under a fixed, subject-independent protocol—and how much EA shifts that ranking.

### 10.4 Protocol comparability rules (for manuscript)

When citing external accuracies, classify comparability explicitly:

| Criterion | This study | Common in literature | Impact on comparison |
|-----------|------------|----------------------|----------------------|
| Split | 80/20 hold-out, subject-disjoint | LOSO, 10-fold CV, 50/50 trial split, within-subject CV | Different splits → different accuracy scale |
| Classes | Binary L/R | BCI-2a: 4 classes (L/R/feet/tongue) | 4-class tasks are harder or not directly comparable |
| EA | Ablated (on/off) | Often absent or only in transfer-learning papers | Must note whether reference used alignment |
| Preprocessing | Shared pipeline, MOABB + custom cache | Paper-specific filters, channel subsets, GAN aug. | Inflated external numbers if preprocessing unequal |
| Aggregation | Trial-level TEST metrics | Sometimes subject-mean or session-mean | Affects reported central tendency |

**Manuscript rule:** include external numbers in the comparison table, but mark each row `Comparable: Yes / Partial / No` and explain in Discussion.

---

## 11. Literature Comparison Table

**Source:** PDF review of Zotero library (`literature_comparison/pdfs/`, extracted June 2026).  
**Our reference row:** hold-out 80/20, binary L/R, EA enabled (unless noted).

### 11.1 PhysioNet MI

| Reference | Model | Reported accuracy | Protocol (from paper) | EA / alignment | Comparable to this study |
|-----------|-------|------------------:|-----------------------|----------------|--------------------------|
| **This study** | CSP+SVM | **70.91%** | 80/20 subject hold-out; binary L/R; 108 subj | Yes | — |
| **This study** | FBCSP+LDA | **70.30%** | same | Yes | — |
| **This study** | EEGNet | **68.18%** | same | Yes | — |
| **This study** | EEGMeModel | **64.65%** | same | Yes | — |
| Kim et al. 2016 | SUTCCSP + RF | 80.05 ± 2.10% | 105 subj; binary L/R; 5-fold CV × 30 iter **per subject**; 24 "significant" subj subset; 14 channels | No | **Partial** — same dataset & binary task; within-subject CV, not subject hold-out |
| Pan et al. 2026 | DCA-SCRCNet | 83.2% mean (92.7% max per subject) | 103 subj; binary L/R; **10-fold CV** multi-subject | Not stated | **Partial** — same dataset; split differs |
| Pan et al. 2026 | DCA-SCRCNet | 76.2% | 4-class PhysioNet | Not stated | **No** — 4-class task |
| Das et al. 2025 | CNN+LSTM hybrid | 96.06% | Binary L/R; GAN augmentation; ROI channels; subject-disjoint split not documented | No | **No** — likely trial-level or inflated split; not reproducible under our protocol |

**PhysioNet interpretation for manuscript:** Our best result (CSP+SVM 70.9% with EA) is **below** Kim et al. 80% and Pan et al. 83%, but those papers use **within-subject or trial-level CV**, which typically yields higher accuracy than subject-disjoint hold-out on 108 subjects. Our result is **plausible** for cross-subject generalization on a large, heterogeneous cohort. Classical methods outperform deep models here, consistent with Kim et al. (CSP-family features) and contrary to Das et al. (96%) which should not be treated as a cross-subject benchmark.

### 11.2 BNCI2014-001 / BCI Competition IV-2a

| Reference | Model | Reported accuracy | Protocol (from paper) | Classes | Comparable to this study |
|-----------|-------|------------------:|-----------------------|---------|--------------------------|
| **This study** | EEGNet | **84.03%** | 80/20 subject hold-out; binary L/R; 9 subj | 2 | — |
| **This study** | EEGMeModel | **71.35%** | same | 2 | — |
| **This study** | CSP+SVM | **63.37%** | same | 2 | — |
| **This study** | FBCSP+LDA | **62.33%** | same | 2 | — |
| Altaheri et al. 2023 | ATCNet | 85.38% (SD) / **70.97%** (LOSO) | BCI-2a; 50/50 trial split (SD); **LOSO** (SI) | 4 | **Partial** — LOSO 70.97% is closest cross-subject reference; 4-class ≠ binary |
| Pan et al. 2026 | DCA-SCRCNet | 90.5% (SD) / **70.7%** (LOSO) | BCI-2a; LOSO for subject-independent | 4 | **Partial** — LOSO 70.7% ≈ our EEGMeModel 71.4% (binary hold-out) |
| Zhang et al. 2026 | NCD-PMSTCNN | 86.24% (SD) / **78.30%** (LOSO) | BCI-2a; LOSO cross-subject | 4 | **Partial** — LOSO upper bound context |
| Singh et al. 2025 | SWCNet | 97.42% (SD) / **61.27%** (LOSO avg) | BCI-2a; subject-dependent vs LOSO | 4 | **Partial** — illustrates SD vs SI gap |
| Liu/Deng et al. 2021 | TSGL-EEGNet | 78.96–81.34% | 5-fold average-validation **per subject** | 4 | **Partial** — within-subject; compares EEGNet vs FBCSP |
| Xiong et al. 2025 | TL + attention | 73.7% avg | **Cross-session** (same subject), not cross-subject | 4 | **No** — session transfer ≠ subject hold-out |

**BNCI interpretation for manuscript:** EEGNet at **84.0%** (binary, subject hold-out, EA) exceeds most published **LOSO 4-class** results (61–78%), which is expected because (a) binary task is easier, (b) hold-out 80/20 retains more training subjects than LOSO (8/9), and (c) BNCI has only 9 subjects with high within-dataset homogeneity. Our EEGMeModel (**71.4%**) aligns with Pan 2026 LOSO (**70.7%**) and Altaheri 2023 LOSO (**70.97%**) despite protocol differences—suggesting our custom architecture performs in the cross-subject deep-learning range but below tuned EEGNet on this small dataset.

### 11.3 EA and transfer-learning context

| Reference | Contribution | Link to this study |
|-----------|--------------|-------------------|
| He & Wu 2020 | Euclidean Alignment (EA); EA-CSP-LDA vs CSP-LDA under **LOSO** on BCI IV 2a | Theoretical and empirical basis for EA ablation; cite in Methods §4.1 step 5 |
| She et al. 2023 | Wasserstein domain adaptation; compares against EA-CSP-LDA | Related Work: cross-subject deep transfer vs simple EA preprocessing |
| Saibene et al. 2024 | Systematic review of DL for MI | Related Work: accuracy ranges 50–96% depending on protocol; justify careful comparison |

**EA interpretation for manuscript:** EA improves **every** model–dataset pair in this benchmark. Largest gain: FBCSP+LDA on PhysioNet (+11.61 pp). This supports He & Wu (2019): alignment reduces inter-subject covariance shift and benefits classical pipelines most on the larger, more heterogeneous PhysioNet cohort.

---

## 12. Manuscript-Ready Framing (Introduction, Related Work, Discussion)

### 12.1 Suggested contribution statement

> We present a reproducible, subject-independent benchmark of four MI decoders—FBCSP+LDA, CSP+SVM, EEGNet, and EEGMeModel—on PhysioNet MI (108 subjects, 64 channels) and BNCI2014-001 (9 subjects, 22 channels), with binary left-/right-hand classification, identical preprocessing, and an Euclidean Alignment ablation. Unlike most prior studies that report a single architecture on one dataset with protocol-specific tuning, we isolate the effect of **dataset scale**, **model family**, and **EA** under a fixed 80/20 subject hold-out.

### 12.2 Related Work paragraphs (factual bullets for drafting)

**Classical MI decoding.** CSP and FBCSP remain strong baselines for MI-EEG (Ramoser et al. 2000; Ang et al. 2008). Kim et al. (2016) reported up to 80% on PhysioNet with complex CSP variants under per-subject cross-validation. Our PhysioNet results confirm that classical pipelines remain competitive under subject-disjoint evaluation (70.3–70.9% with EA).

**Deep learning MI decoding.** EEGNet (Lawhern et al. 2018) is widely used as a compact CNN baseline. Recent architectures (ATCNet, SWCNet, PMSTCNN, DCA-SCRCNet) report 85–97% on BCI-2a under subject-dependent or within-session protocols, but **61–78% under LOSO** (Altaheri et al. 2023; Singh et al. 2025; Zhang et al. 2026; Pan et al. 2026). Our BNCI EEGNet result (84.0%, binary hold-out) should be discussed alongside these protocol differences.

**Cross-subject alignment.** He & Wu (2020) introduced EA as an efficient preprocessing step before Euclidean-space classifiers. Domain-adaptation networks (She et al. 2023) pursue the same goal with learned feature alignment. We adopt EA as a lightweight, interpretable ablation rather than end-to-end transfer learning.

**Benchmark reproducibility.** MOABB (Jayaram et al. 2018) standardizes dataset access; this project extends that with cached tensors, shared evaluation code, and public repository artifacts.

### 12.3 Discussion points (literature-informed)

1. **Ranking reversal across datasets.** Classical methods win on PhysioNet; EEGNet wins on BNCI. Literature also reports dataset-dependent winners (e.g., Liu et al. 2021: EEGNet vs FBCSP on BCI-2a). Proposed explanation: PhysioNet's larger inter-subject variability and higher channel count favour explicit spatial filtering; BNCI's small homogeneous cohort favours end-to-end CNN feature learning.

2. **EEGMeModel vs EEGNet.** EEGMeModel does not outperform EEGNet on either dataset. The value is **negative result under fair conditions**: a CNN+Transformer architecture with more capacity does not automatically improve over EEGNet when preprocessing and splits are held constant. Compare to Pan et al. 2026 where specialized attention modules beat reproduced EEGNet (81.6% SD on BCI-2a).

3. **Contextualizing absolute accuracies.** PhysioNet 65–71% (cross-subject, EA) is consistent with the lower end of review-reported ranges (Saibene et al. 2024) for subject-independent evaluation. BNCI 84% (EEGNet) is high relative to LOSO 4-class literature but reflects binary task + 80/20 split on 9 subjects; generalization to new cohorts should be stated cautiously.

4. **Why not claim SOTA.** Published SOTA on BCI-2a (86–97%) overwhelmingly uses subject-dependent splits or 4-class protocols. Our study prioritizes **fair cross-model comparison** over maximizing a single accuracy number.

5. **Limitations vs literature.** (a) Single hold-out seed (`random_state=42`); literature often uses LOSO or multiple seeds. (b) BNCI n=9 limits statistical power compared to PhysioNet n=108. (c) Binary-only paradigm excludes feet/tongue classes used in BCI Competition benchmarks.

### 12.4 Suggested manuscript table caption

> **Table X.** Comparison with selected prior MI studies. Rows marked *Partial* or *No* differ in evaluation protocol, class count, or preprocessing from this study (80/20 subject-disjoint hold-out, binary L/R, shared pipeline, EA on). Accuracies from original papers; see `literature_comparison/LITERATURE_COMPARISON.md` for extraction notes.

---

## 13. Per-Class Metrics (BNCI, hold-out, with EA)

### EEGNet + EA — 576 test trials

| Class | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| left_hand | 0.845 | 0.833 | 0.839 | 288 |
| right_hand | 0.836 | 0.847 | 0.841 | 288 |
| **Accuracy** | | | **0.840** | 576 |

### EEGMeModel + EA — 576 test trials

| Class | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| left_hand | 0.727 | 0.684 | 0.705 | 288 |
| right_hand | 0.702 | 0.743 | 0.722 | 288 |
| **Accuracy** | | | **0.714** | 576 |

### FBCSP+LDA + EA — 576 test trials

| Class | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| left_hand | 0.598 | 0.750 | 0.666 | 288 |
| right_hand | 0.665 | 0.497 | 0.569 | 288 |
| **Accuracy** | | | **0.623** | 576 |

### CSP+SVM + EA — 576 test trials

Metrics available in `outputs/csp_svm_ea_holdout/metrics.json` and `classification_report.txt`.

---

## 14. Empirical Observations (data-derived)

1. Model ranking order differs between PhysioNet and BNCI when EA is enabled (see §9.3).
2. EA increases accuracy on every model–dataset combination in this benchmark.
3. Largest EA-induced accuracy gain: FBCSP+LDA on PhysioNet (+11.61 pp).
4. Smallest EA-induced accuracy gain: EEGNet and EEGMeModel on BNCI (+2.43 pp each).
5. Highest single result: EEGNet + EA on BNCI (84.03%, κ = 0.681).
6. Highest result on PhysioNet: CSP+SVM + EA (70.91%, κ = 0.418).
7. FBCSP+LDA on BNCI exhibits asymmetric recall: left_hand 0.750, right_hand 0.497 (with EA).
8. EEGNet and EEGMeModel on BNCI without EA (81.60% and 68.92%) exceed their PhysioNet counterparts with EA (68.18% and 64.65%).
9. Classical methods (FBCSP+LDA, CSP+SVM) on PhysioNet with EA (70.3%, 70.9%) exceed all deep models on the same dataset.
10. Evaluation uses a single fixed hold-out partition (`random_state=42`).
11. PhysioNet classical results (70.3–70.9% with EA) are below Kim et al. 2016 (80.05%, within-subject CV) and Pan et al. 2026 (83.2% mean, 10-fold CV), consistent with stricter subject-disjoint evaluation (§11.1).
12. BNCI EEGMeModel (71.35% with EA) is numerically close to Pan et al. 2026 LOSO (70.7%) and Altaheri et al. 2023 LOSO (70.97%) on BCI-2a, despite protocol differences (§11.2).
13. BNCI EEGNet (84.03% with EA) exceeds published LOSO 4-class deep-learning range (61–78%) but uses binary classes and 80/20 hold-out; not directly comparable to subject-dependent SOTA (86–97%) (§11.2, §12.3).

---

## 15. Bibliography

### Core methods and datasets (cited in this benchmark)

1. Schalk, G., et al. (2004). BCI2000: a general-purpose brain-computer interface (BCI) system. *IEEE Transactions on Biomedical Engineering*.
2. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*.
3. Tangermann, M., et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*.
4. Jayaram, V., et al. (2018). MOABB: trustworthy algorithm benchmarking for BCIs. *Journal of Neural Engineering*.
5. Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013. DOI: 10.1088/1741-2552/aace8c
6. Ang, K. K., et al. (2008). Filter bank common spatial pattern (FBCSP) in brain-computer interface. *IEEE International Conference on Robotics and Automation*.
7. Ramoser, H., Muller-Gerking, J., & Pfurtscheller, G. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. *IEEE Transactions on Rehabilitation Engineering*.
8. Blankertz, B., et al. (2008). Optimizing spatial filters for robust EEG single-trial analysis. *IEEE Signal Processing Magazine*.
9. He, H., & Wu, D. (2020). Transfer learning for brain–computer interfaces: a Euclidean space data alignment approach. *IEEE Transactions on Biomedical Engineering*. DOI: 10.1109/TBME.2019.2913914
10. Pfurtscheller, G., & Neuper, C. (2001). Motor imagery and direct brain-computer communication. *Proceedings of the IEEE*.

### Literature comparison sources (reviewed from Zotero, June 2026)

11. Kim, Y., et al. (2016). Motor imagery classification using mu and beta rhythms with SUTCCSP. *Computational Intelligence and Neuroscience*. DOI: 10.1155/2016/1489692
12. Schirrmeister, R. T., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*. DOI: 10.1002/hbm.23730
13. Liu, K., et al. / Deng, X., et al. (2021). Advanced TSGL-EEGNet for motor imagery EEG-based BCIs. *IEEE Access*. DOI: 10.1109/ACCESS.2021.3056088
14. Altaheri, H., et al. (2023). Physics-informed attention temporal convolutional network for EEG-based motor imagery classification. *IEEE Transactions on Industrial Informatics*. DOI: 10.1109/TII.2022.3197419
15. She, Q., et al. (2023). Improved domain adaptation network based on Wasserstein distance for motor imagery EEG classification. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*. DOI: 10.1109/TNSRE.2023.3241846
16. Saibene, A., et al. (2024). Deep learning in motor imagery EEG signal decoding: a systematic review. *Neurocomputing*. DOI: 10.1016/j.neucom.2024.128577
17. Singh, K., et al. (2025). A novel CNN with sliding window technique for enhanced classification of MI-EEG sensor data. *IEEE Sensors Journal*. DOI: 10.1109/JSEN.2024.3515252
18. Das, A., et al. (2025). Enhanced EEG signal classification in BCIs using hybrid deep learning models. *Scientific Reports*. DOI: 10.1038/s41598-025-07427-2
19. Xiong, C., et al. (2025). Cross-session EEG motor imagery classification method based on transfer learning. *ICPECA*. DOI: 10.1109/ICPECA63937.2025.10928860
20. Zhang, N., et al. (2026). A partitioned multi-scale spatiotemporal CNN for motor imagery-EEG decoding. *Biomedical Signal Processing and Control*. DOI: 10.1016/j.bspc.2025.109234
21. Pan, J., et al. (2026). Low-redundancy motor imagery EEG decoding based on dynamic attention and feature reconstruction. *Neurocomputing*. DOI: 10.1016/j.neucom.2025.132004
22. Hameed, I., et al. (2025). Enhancing motor imagery EEG signal decoding through machine learning: a systematic review. *Computers in Biology and Medicine*. DOI: 10.1016/j.compbiomed.2024.109534

---

## 16. Output Artifacts Index

| Path | Contents |
|------|----------|
| `outputs/pipeline_comparison.csv` | All hold-out results (both datasets, all models, EA on/off) |
| `outputs/pipeline_comparison.json` | Same data, JSON format |
| `outputs/eegme_ea_holdout/` | EEGMeModel + EA: `metrics.json`, `predictions.csv`, `classification_report.txt`, `confusion_matrix.png`, `training_history.csv`, `checkpoints/best.pt` |
| `outputs/eegme_no_ea_holdout/` | EEGMeModel, no EA |
| `outputs/eegnet_ea_holdout/` | EEGNet + EA |
| `outputs/eegnet_no_ea_holdout/` | EEGNet, no EA |
| `outputs/lda_ea_holdout/` | FBCSP+LDA + EA |
| `outputs/lda_no_ea_holdout/` | FBCSP+LDA, no EA |
| `outputs/csp_svm_ea_holdout/` | CSP+SVM + EA |
| `outputs/csp_svm_no_ea_holdout/` | CSP+SVM, no EA |
| `data/processed/physionet_lr_ea_108sub_all/meta.json` | PhysioNet cache metadata (subjects, channels, dimensions) |
| `configs/default.yaml` | Full hyperparameter specification |
| `src/physionet_mi/models/eegme.py` | EEGMeModel architecture source |
| `src/physionet_mi/models/eegnet.py` | EEGNet architecture source |
| `src/physionet_mi/baseline/lda_fbcsp.py` | FBCSP+LDA implementation |
| `src/physionet_mi/baseline/csp_svm_classifier.py` | CSP+SVM implementation |

---

## 17. Companion Files to Upload Alongside This Document

| Priority | File | Role |
|----------|------|------|
| Required | `PAPER_MANUSCRIPT_BRIEF.md` (this file) | Complete experimental record |
| Required | `outputs/pipeline_comparison.csv` | Machine-readable results table |
| Required | `configs/default.yaml` | All hyperparameters |
| Required | `data/processed/physionet_lr_ea_108sub_all/meta.json` | PhysioNet subject split and dimensions |
| Recommended | `outputs/eegnet_ea_holdout/classification_report.txt` | Per-class metrics (BNCI best model) |
| Recommended | `outputs/lda_ea_holdout/classification_report.txt` | Per-class metrics (PhysioNet classical baseline) |
| Recommended | `outputs/eegme_ea_holdout/training_history.csv` | Training convergence data |
| Recommended | `outputs/*/confusion_matrix.png` | Confusion matrices (figures) |
| Optional | `src/physionet_mi/models/eegme.py` | Architecture details |
| Optional | `src/physionet_mi/models/eegnet.py` | Architecture details |
| Optional | `src/physionet_mi/baseline/lda_fbcsp.py` | Baseline pipeline details |
| Optional | `src/physionet_mi/baseline/csp_svm_classifier.py` | Baseline pipeline details |
| Optional | `outputs/pipeline_comparison.json` | Results in JSON |
| Recommended | `literature_comparison/LITERATURE_COMPARISON.md` | Literature extraction and comparability notes |
| Recommended | `literature_comparison/comparison_data.json` | Structured literature data |
| Optional | `literature_comparison/pdfs/*.pdf` | Source PDFs (12 papers from Zotero) |

---

*Generated from completed hold-out experiments. Literature sections added from Zotero PDF review. Last updated: June 2026.*
