# PhysioNet MI — Left vs Right (EEGMeModel)

Proyecto Python para clasificar **mano izquierda / mano derecha** en PhysioNet MI (MOABB), usando el modelo profundo `EEGMeModel` (CNN local + Transformer). Incluye el mismo preprocesado del taller (`workshop.ipynb`), evaluación **hold-out 80/20** y **LOSO**, y baseline **FBCSP + LDA**.

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

### 1. Preparar datos (descarga + caché)

```bash
python scripts/prepare_data.py --config configs/preprocess_no_ea.yaml
python scripts/prepare_data.py --config configs/preprocess_ea.yaml
```

Genera arrays en `data/processed/physionet_lr_{ea|no_ea}_*/holdout/`.

### 2. Baseline LDA (sanity check del taller)

```bash
python scripts/run_baseline_lda.py --config configs/preprocess_ea.yaml
```

### 3. Entrenar red — hold-out

```bash
python scripts/train_holdout.py --config configs/preprocess_no_ea.yaml
python scripts/train_holdout.py --config configs/preprocess_ea.yaml --run-name dl_ea_holdout
```

### 4. Entrenar red — LOSO

```bash
# Desarrollo rápido (3 folds)
python scripts/train_loso.py --config configs/preprocess_no_ea.yaml --max-folds 3

# Cohorte completa (~109 folds, muy lento en CPU)
python scripts/train_loso.py --config configs/preprocess_ea.yaml
```

### 5. Ablation completa

```bash
python scripts/run_ablation.py --max-folds 3
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
  models/        # EEGMeModel
  datasets/      # PyTorch Dataset
  training/      # hold-out, LOSO, trainer
  baseline/      # FBCSP + LDA
  evaluation/    # métricas y reportes
  cli/           # entry points
```

## Referencia

- Tarea y preprocesado: `workshop.ipynb`
- Modelo original: `funtions.py` / `code.ipynb` (dataset distinto; no usar para PhysioNet sin adaptar)
