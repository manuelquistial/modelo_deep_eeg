# Comparación con literatura — extracción desde PDFs de Zotero

**Fecha de extracción:** 2026-06-05  
**Fuente:** 12 PDFs copiados desde `~/Zotero/storage/`  
**Texto extraído:** `extracted_text/` (pdftotext)  
**PDFs:** `pdfs/`

---

## Advertencia metodológica (leer antes de usar los números)

Tus resultados usan **hold-out 80/20 por sujetos**, **MI binario (L/R)**, y **preprocesado idéntico** para 4 modelos + ablation EA.

La mayoría de papers extraídos usan protocolos distintos:

| Diferencia frecuente | Impacto |
|---|---|
| **4 clases** (BNCI estándar) vs tu **2 clases** | Accuracy suele ser más alta o no comparable |
| **LOSO** vs tu **hold-out 80/20** | LOSO es más duro; hold-out puede inflar o deflacionar según cohorte |
| **CV within-subject** vs **split por sujetos** | Within-subject mucho más optimista |
| **Sin EA** vs tu **ablation EA** | No comparable directamente sin alinear preprocesado |
| **GAN augmentation / channel selection** | Puede inflar accuracy (ej. Das 2025 → 96%) |

**Regla:** solo comparar en el manuscrito cuando **dataset + clases + split + preprocesado** estén alineados, o explicar explícitamente la diferencia.

---

## Tus resultados de referencia (hold-out 80/20 + EA)

| Dataset | CSP+SVM | FBCSP+LDA | EEGNet | EEGMeModel |
|---|---:|---:|---:|---:|
| PhysioNet (108 subj, 64 ch) | **70.9%** | 70.3% | 68.2% | 64.6% |
| BNCI2014-001 (9 subj, 22 ch) | 63.4% | 62.3% | **84.0%** | 71.4% |

---

## Tier 1 — Papers para tabla de comparación (extraídos de PDF)

### PhysioNet MI — binario o multi-clase

| Paper | Modelo | Accuracy reportada | Protocolo (del PDF) | ¿Comparable? |
|---|---|---:|---|---|
| **Pan et al. 2026** | DCA-SCRCNet 2-class | 92.7% máx / **83.2% media** (103 subj) | 10-fold CV multi-sujeto | Parcial — mismo dataset, split distinto |
| **Kim et al. 2016** | SUTCCSP + RF | **80.05% ± 2.10%** (24 "significant" subj) | 5-fold CV × 30 iter **por sujeto** | Parcial — binario L/R, CV within-subject |
| **Das et al. 2025** | CNN+LSTM hybrid | **96.06%** | Split no documentado como subject-disjoint; GAN aug. | **Baja** — número sospechosamente alto |

### BNCI2014-001 / BCI Competition IV-2a

| Paper | Modelo | Accuracy reportada | Protocolo (del PDF) | ¿Comparable? |
|---|---|---:|---|---|
| **Pan et al. 2026** | DCA-SCRCNet | **90.5%** (SD) / **70.7%** (SI/LOSO) | 4 clases; LOSO para SI | Parcial — LOSO SI ~71% vs tu 71–84% binario hold-out |
| **Altaheri et al. 2023** | ATCNet | **85.38%** (SD) / **70.97%** (LOSO) | 4 clases; 50/50 trials SD; LOSO SI | **Alta** para cross-subject (LOSO 70.97%) |
| **Zhang et al. 2026** | NCD-PMSTCNN | **86.24%** (SD) / **78.30%** (LOSO) | 4 clases; LOSO cross-subject | Parcial — LOSO referencia |
| **Singh et al. 2025** | SWCNet | **97.42%** (SD) / **61.27%** (LOSO avg) | 4 clases; SD 50/30/20; LOSO SI | Parcial — LOSO ~61% |
| **Liu/Deng et al. 2021** | TSGL-EEGNet | **78.96–81.34%** | 4 clases; 5-fold avg-validation **por sujeto** | Parcial — within-subject |
| **Xiong et al. 2025** | TL + attention | **73.7%** avg | 4 clases; **cross-session** (no LOSO) | Baja — cross-session ≠ tu protocolo |

---

## Tier 2 — Papers metodológicos (citar, no tabla numérica directa)

| Paper | Uso en tu manuscrito |
|---|---|
| **He & Wu 2020 (EA-TBME)** | Fundamento de **Euclidean Alignment**; EA-CSP-LDA con **LOSO** en BCI IV 2a; justifica tu ablation EA |
| **Schirrmeister et al. 2017** | Contexto **deep learning + EEG**; Deep ConvNet como referencia histórica |
| **She et al. 2023** | **Domain adaptation** cross-subject en BNCI; cita EA-CSP-LDA como baseline |
| **Saibene et al. 2024** | **Systematic review** — mapa del campo, rangos de accuracy heterogéneos |

---

## Detalle por paper (extraído del PDF)

### 1. Pan et al. 2026 — `Pan2026_LowRedundancy_MI.pdf`
- **DOI:** 10.1016/j.neucom.2025.132004
- **Datasets:** BCI-2a + PhysioNet
- **Clases:** 4 (BNCI) / 2 y 4 (PhysioNet)
- **Protocolo:** LOSO para SI en BCI-2a; 10-fold CV en PhysioNet
- **Números clave:**
  - BCI-2a SD: **90.5%** (κ=0.87); SI/LOSO: **70.7%** (κ=0.61)
  - PhysioNet 2-class: **92.7%** máx, media **83.2%** en 103 sujetos
  - PhysioNet 4-class: **76.2%**
  - EEGNet reproducido BCI-2a SD: **81.6%**
- **Para tu paper:** el paper más alineado (ambos datasets). Comparar SI/LOSO 70.7% con tu enfoque cross-subject.

### 2. Das et al. 2025 — `Das2025_EnhancedEEG_PhysioNet.pdf`
- **DOI:** 10.1038/s41598-025-07427-2
- **Dataset:** PhysioNet MI
- **Clases:** mano L vs mano R
- **Protocolo:** GAN augmentation, ROI channels; split por sujetos no claro
- **Números clave:** RF 91.04%, CNN 88.18%, **CNN+LSTM 96.06%**
- **Para tu paper:** citar con cautela; discutir por qué 96% ≠ comparable con hold-out subject-disjoint.

### 3. Liu/Deng et al. 2021 — `Liu2021_TSGL_EEGNet.pdf`
- **DOI:** 10.1109/ACCESS.2021.3056088
- **Dataset:** BCI Competition IV-2a (+ IIIa)
- **Clases:** 4
- **Protocolo:** 5-fold average-validation por sujeto; compara EEGNet vs **FBCSP**
- **Números clave:** TSGL-EEGNet **78.96%** (κ=0.7194); stacking **81.34%** (κ=0.7511)
- **Para tu paper:** referencia EEGNet/FBCSP en BNCI; protocolo within-subject.

### 4. Singh et al. 2025 — `Singh2025_CNN_SlidingWindow.pdf`
- **DOI:** 10.1109/JSEN.2024.3515252
- **Dataset:** BCIC-IV-2a (4 clases)
- **Protocolo:** SD 50/30/20; SI setup I (mezcla sujetos); SI setup II **LOSO**
- **Números clave:** SD **97.42%**; SI setup I **91.89%**; LOSO avg **61.27%**
- **Para tu paper:** contraste SD optimista vs LOSO ~61%.

### 5. Zhang et al. 2026 — `Zhang2026_PMSSTCNN.pdf`
- **DOI:** 10.1016/j.bspc.2025.109234
- **Dataset:** BCI IV-2a
- **Protocolo:** SD competition split; **LOSO** cross-subject
- **Números clave:** SD **86.24%** (κ=81.68); LOSO **78.30%**; Deep ConvNet SD 73.15%
- **Para tu paper:** SOTA reciente; LOSO 78.3% como referencia cross-subject.

### 6. Altaheri et al. 2023 — `Altaheri2023_PIATCN.pdf`
- **DOI:** 10.1109/TII.2022.3197419
- **Dataset:** BCI-2a
- **Protocolo:** SD 50/50 trials; SI **LOSO**
- **Números clave:** ATCNet SD **85.38%** (κ=0.81); SI **70.97%**
- **Para tu paper:** referencia sólida; LOSO 70.97% cercano a tu rango BNCI.

### 7. She et al. 2023 — `She2023_DomainAdaptation.pdf`
- **DOI:** 10.1109/TNSRE.2023.3241846
- **Dataset:** BCI IV 2a y 2b
- **Protocolo:** LOSO; 1 sujeto target (sesión 2), 8 source (sesión 1)
- **Números clave:** mejora ~2% vs DRDA/DAFS; compara con **EA-CSP-LDA**
- **Para tu paper:** related work transfer learning; no extraer número único sin Table II manual.

### 8. Xiong et al. 2025 — `Xiong2025_CrossSession_TL.pdf`
- **DOI:** 10.1109/ICPECA63937.2025.10928860
- **Dataset:** BCI-2a
- **Protocolo:** **Cross-session** (mismo sujeto)
- **Números clave:** propuesto **73.7%** (κ=65.6%); EEGNet **69.2%**
- **Para tu paper:** diferenciar cross-session vs cross-subject.

### 9. Kim et al. 2016 — `Kim2016_MuBeta_CSP_PhysioNet.pdf`
- **DOI:** 10.1155/2016/1489692
- **Dataset:** PhysioNet (105 subj tras exclusiones)
- **Clases:** L/R mano
- **Protocolo:** 5-fold CV por sujeto; subset 24 "significant" subjects
- **Números clave:** SUTCCSP+RF **80.05%** (24 subj); 14 canales seleccionados
- **Para tu paper:** baseline clásico CSP en PhysioNet; protocolo distinto.

### 10. He & Wu 2020 — `He2020_EuclideanAlignment_TBME.pdf`
- **DOI:** 10.1109/TBME.2019.2913914
- **Dataset:** BCI IV Dataset 1 y 2a
- **Protocolo:** **LOSO**; EA-CSP-LDA vs CSP-LDA vs RA-MDRM
- **Números clave:** EA-CSP-LDA supera CSP-LDA en 14/16 sujetos; mejor que RA-MDRM en 11/16
- **Para tu paper:** **cita obligatoria** para ablation EA.

### 11. Schirrmeister et al. 2017 — `Schirrmeister2017_DeepCNN_EEG.pdf`
- **DOI:** 10.1002/hbm.23730
- **Uso:** related work deep CNN para EEG

### 12. Saibene et al. 2024 — `Saibene2024_DL_MI_Review.pdf`
- **DOI:** 10.1016/j.neucom.2024.128577
- **Uso:** systematic review; rangos 50–96% según protocolo

---

## Papers canónicos que faltan en tu Zotero (añadir manualmente)

| Paper | Por qué |
|---|---|
| Lawhern et al. 2018 — EEGNet | Referencia original de tu modelo baseline |
| Jayaram et al. — MOABB | Usas MOABB para cargar datos |
| Schalk et al. — PhysioNet EEGBCI | Descripción oficial del dataset |
| Tangermann et al. — BCI Competition IV review | Descripción oficial BNCI |
| Blankertz et al. — FBCSP | Baseline clásico |
| Lotte et al. — review spatial filters | Contexto métodos clásicos |

---

## Estructura de carpetas

```
literature_comparison/
├── LITERATURE_COMPARISON.md    ← este archivo
├── comparison_data.json        ← datos estructurados (máquina)
├── pdfs/                       ← 12 PDFs copiados de Zotero
│   ├── Pan2026_LowRedundancy_MI.pdf
│   ├── Das2025_EnhancedEEG_PhysioNet.pdf
│   ├── ...
└── extracted_text/             ← texto extraído con pdftotext
    ├── Pan2026_LowRedundancy_MI.txt
    └── ...
```

---

## Próximo paso sugerido para el manuscrito

Armar **Table: Literature comparison** con columnas:

`Reference | Dataset | Classes | Channels | Subjects | Split | Preprocessing (EA?) | Best model | Accuracy | Comparable (Y/N/Partial)`

Rellenar solo filas con **Partial** o **Y** y una nota de diferencia en la Discussion.
