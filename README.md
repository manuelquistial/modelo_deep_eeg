# PhysioNet MI — Left vs Right (EEGMeModel)

Proyecto Python para clasificar **mano izquierda / mano derecha** en PhysioNet MI (MOABB). Incluye:

| Modelo | Tipo | Script |
|--------|------|--------|
| **EEGMeModel** | DL (CNN + Transformer) | `train_holdout.py` / `train_loso.py` |
| **EEGNet** | DL (Lawhern 2018, desde `modelo_bilstm`) | `--config configs/model_eegnet.yaml` |
| **FBCSP + LDA** | Baseline clásico (taller) | `run_baseline_lda.py` |
| **CSP + SVM** | Baseline clásico (Sun et al. / `modelo_bilstm`) | `run_baseline_csp_svm.py` |

Evaluación **hold-out 80/20** (sujetos) y **LOSO**. Ablation compara todos los modelos.

## Requisitos

- **Python 3.11+** (Paperspace Gradient usa 3.11; macOS local: 3.12.6 recomendado)
- **PyTorch** vía extra `[ml]` (en Mac Intel: `torch 2.2.x` + `numpy<2`)
- Datos MOABB/MNE (~109 sujetos; primera descarga puede tardar)

### Compatibilidad PyTorch / NumPy / MOABB

| Plataforma | Python | PyTorch | NumPy |
|------------|--------|---------|-------|
| macOS Intel (x86_64) | 3.12 | 2.2.x | 1.26.x |
| macOS Apple Silicon | 3.12+ | 2.6+ | 2.x |
| Linux / Windows | 3.12+ | 2.6+ | 2.x |

MOABB declara `numpy>=2`, pero en Mac Intel convive con `numpy 1.26` y PyTorch 2.2. El proyecto fija `numpy>=1.26,<2` por esa razón.

### Nota Python 3.14

PyTorch aún no publica wheels para 3.14. Usa 3.12 hasta que estén disponibles.

### Paperspace Gradient (GPU)

Gradient suele tener driver CUDA **12.4**. Si `torch.cuda.is_available()` da `False` tras `pip install -e ".[ml,dev]"`, reinstala PyTorch con CUDA 12.4 **antes** o **después** del resto:

```bash
source .venv/bin/activate
pip install -e ".[dev]"   # sin torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Instala el resto de dependencias ML si hace falta: `pip install -e .` (editable, sin reinstalar torch).

## Instalación

```bash
cd modelo_deep_eeg
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[ml,dev]"
```

### Smoke test (10 sujetos, rápido)

```bash
python scripts/prepare_data.py --config configs/smoke_test.yaml
python scripts/run_baseline_lda.py --config configs/smoke_test.yaml
python scripts/train_holdout.py --config configs/smoke_test.yaml
python scripts/train_loso.py --config configs/smoke_test.yaml --max-folds 2
```

## Configuración

- `configs/default.yaml` — parámetros base
- `configs/preprocess_no_ea.yaml` — sin Euclidean Alignment
- `configs/preprocess_ea.yaml` — con EA (como el taller)

Variables de entorno opcionales:

- `PHYSIONET_MI_DATA_DIR` — sobrescribe `data.mne_data_dir`

## Uso

### Datasets soportados

| Dataset | Config base | Sujetos | Canales | Hz |
|---------|-------------|---------|---------|-----|
| **PhysioNet MI** | `configs/preprocess_ea.yaml` / `preprocess_no_ea.yaml` | ~108 | 64 | 160 |
| **BNCI2014_001** (modelo_bilstm) | `configs/preprocess_bnci_ea.yaml` / `preprocess_bnci_no_ea.yaml` | 9 | 22 | 125 |

En `configs/default.yaml`: `data.dataset: physionet | bnci2014_001`

### 1. Preparar datos

```bash
# PhysioNet (ea y no_ea)
python scripts/prepare_data.py --config configs/preprocess_ea.yaml
python scripts/prepare_data.py --config configs/preprocess_no_ea.yaml

# BNCI2014_001 (MOABB; primera vez descarga)
python scripts/prepare_data.py --config configs/preprocess_bnci_ea.yaml
python scripts/prepare_data.py --config configs/preprocess_bnci_no_ea.yaml
```

Genera caché en `artifacts/cache/{dataset}_lr_{ea|no_ea}_*/holdout/` (ver `artifacts/README.md`).

Tras clonar o si vienes de una versión antigua:

```bash
python scripts/setup_artifacts_layout.py
```

### 2. Baselines clásicos

```bash
# FBCSP + LDA (taller)
python scripts/run_baseline_lda.py --config configs/preprocess_ea.yaml

# CSP + SVM (modelo_bilstm — grid search SVM RBF)
python scripts/run_baseline_csp_svm.py --config configs/preprocess_ea.yaml
# BNCI baselines
python scripts/run_baseline_csp_svm.py --config configs/preprocess_bnci_ea.yaml
python scripts/run_baseline_lda.py --config configs/preprocess_bnci_ea.yaml
```

### 3. Entrenar red — hold-out

```bash
# EEGMeModel (default)
python scripts/train_holdout.py --config configs/preprocess_no_ea.yaml

# EEGNet
python scripts/train_holdout.py --config configs/model_eegnet.yaml
python scripts/train_holdout.py --config configs/model_eegnet_ea.yaml
```

### 4. Entrenar red — LOSO

```bash
python scripts/train_loso.py --config configs/preprocess_no_ea.yaml --max-folds 3
python scripts/train_loso.py --config configs/model_eegnet.yaml --max-folds 3
# EEGNet BNCI
python scripts/train_holdout.py --config configs/model_eegnet_bnci.yaml
```

### 5. Comparar pipelines × datasets × preprocesamiento

Todos los modelos (**EEGMe**, **EEGNet**, **FBCSP+LDA**, **CSP+SVM**) consumen la **misma caché** generada por `prepare_data` con el pipeline de `preprocessing.py`:

1. Eliminación de outliers (800 µV)
2. Filtro pasa-alto (4 Hz)
3. Harmonización temporal
4. Euclidean Alignment (opcional: `ea` / `no_ea`)

```bash
# Comparación completa: 2 datasets × 2 preprocess × 4 modelos (hold-out)
python scripts/compare_datasets.py --skip-loso

# Incluir LOSO (lento)
python scripts/compare_datasets.py --max-folds 3

# Solo BNCI, ambos preprocess
python scripts/compare_datasets.py --datasets bnci2014_001 --preprocess ea,no_ea --skip-loso

# Solo EA en ambos datasets
python scripts/compare_datasets.py --preprocess ea --skip-loso
```

Salida: `outputs/pipeline_comparison.csv` y `.json`

Columnas clave: `dataset`, `preprocess` (`ea`|`no_ea`), `model`, `protocol`, `accuracy`, …

### 6. Ablation (mismo motor que compare_datasets)

```bash
# PhysioNet, ea + no_ea (default ablation)
python scripts/run_ablation.py --max-folds 3 --skip-loso

# Ambos datasets
python scripts/run_ablation.py --datasets physionet,bnci2014_001 --skip-loso
```

Resultados en `outputs/ablation_summary.csv`.

## Salidas

Cada run guarda en `outputs/<run_name>/`:

- `metrics.json`, `classification_report.txt`, `confusion_matrix.png`
- `predictions.csv`, `training_history.csv`
- `checkpoints/best.pt`

## Tests

```bash
pytest -q
```

Los tests usan datos sintéticos (no requieren descarga MOABB).

## Estructura

```
src/physionet_mi/
  data/          # MOABB, preprocesado, caché
  models/        # EEGMeModel, EEGNet, registry
  datasets/      # PyTorch Dataset
  training/      # hold-out, LOSO, trainer
  baseline/      # FBCSP+LDA, CSP+SVM
  evaluation/    # métricas y reportes
  cli/           # entry points
```

## Referencia

- Tarea y preprocesado: `workshop.ipynb`
- Modelo original: `funtions.py` / `code.ipynb` (dataset distinto; no usar para PhysioNet sin adaptar)
