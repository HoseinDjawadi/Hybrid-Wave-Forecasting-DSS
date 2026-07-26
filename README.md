# Coastal Wave Forecasting for Maritime Safety: A Causal, Interpretable, and Uncertainty-Aware Decision Support Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the official implementation of the paper:

> **An Expert Decision Support system for Marine Safety Powered by Causal Discovery**  
> *Submitted to Reliability Engineering & System Safety (Elsevier)*

We introduce a  Decision Support System for coastal wave forecasting that combines:
- **Causal discovery (PCMCI+)** to isolate physically meaningful offshore predictors,
- **Structurally constrained, natively interpretable models** (EBM and Attention‑LSTM) that resist the Regularisation Tax,
- **Uncertainty decoupling** (conformal prediction + ensemble disagreement) and **VIKOR multi‑criteria decision making** for operational risk classification.

## Key Contributions

- PCMCI+ reduces 102 offshore grid points to 5 causally validated predictors
- Over‑parameterised models (TFT, LightGBM) collapse at extended horizons; EBM and Attention‑LSTM maintain stable accuracy
- Aleatoric and epistemic uncertainties are formally decoupled
- VIKOR‑based DSS outputs colour‑coded operational risk levels (Safe / Warning / Danger)

## Repository Structure
Hybrid-Wave-Forecasting-DSS/
├── notebooks/ # Jupyter notebooks for each pipeline stage
├── src/ # Reusable Python utility functions
├── data/ # Sample input and data access instructions
├── outputs/ # Generated figures and tables
│ ├── figures/
│ └── tables/
├── models/ # Trained model weights (download links)
├── requirements.txt # Python dependencies
└── README.md

## Installation

```bash
git clone https://github.com/HoseinDjawadi/Hybrid-Wave-Forecasting-DSS.git
cd Hybrid-Wave-Forecasting-DSS
pip install -r requirements.txt
```

## Data
Due to a non‑disclosure agreement, the raw buoy observations cannot be publicly distributed. Researchers interested in accessing the full dataset should contact the corresponding author (see below). A small synthetic sample (data/sample_input.csv) is provided for code testing.

The offshore predictor data are derived from CMEMS / HYCOM reanalysis products. Bathymetric data are from the GEBCO 2024 Grid. Instructions for obtaining these public datasets are in data/README.md.

## Usage
The pipeline is organised as a series of Jupyter notebooks inside notebooks/:

01_Predictor_Selection.ipynb — causal predictor selection with PCMCI+

02_Feature_Engineering.ipynb — lag, rolling, and physics‑informed feature engineering

03_Baselines.ipynb — persistence and SMA‑12h statistical baselines

04_LightGBM.ipynb — LightGBM baseline training and evaluation

05_EBM.ipynb — Explainable Boosting Machine

06_Attention_LSTM.ipynb — Attention‑LSTM model

07_TFT.ipynb — Temporal Fusion Transformer baseline

08_DSS_Dashboard.ipynb — meta‑ensemble, conformal prediction, VIKOR DSS

## Results
All manuscript figures are available in outputs/figures/. Key results include:

Regularisation Tax: TFT R² falls from 0.34 (+3 h) to −0.05 (+24 h); EBM and Attention‑LSTM remain above 0.20

PICP: Conformal prediction intervals achieve 89.99 % empirical coverage (nominal 90 %)

VIKOR Risk: Danger classification correctly isolates the April 2024 storm peak

## Citation
If you use this code or data in your research, please cite our paper:

```bibtex
@article{seyed2025hybrid,
  title   = {Coastal Wave Forecasting for Maritime Safety: A Causal, Interpretable, and Uncertainty-Aware Decision Support Framework},
  author  = {M.R. Nikoo, M.H. Seyed‑Djawadi and Talal Etri},
  journal = {Reliability Engineering & System Safety},
  year    = {2026},
  publisher = {Elsevier}
}
```
## License
This project is licensed under the MIT License — see LICENSE for details.

## Contact
For data access requests or questions: t.etri1@squ.edu.om

---

## 6 – data

```markdown
# Data Access

## Buoy Observations

The nearshore buoy observations used in this study are subject to a non‑disclosure agreement and **cannot** be publicly redistributed. Researchers may request access by contacting the corresponding author at [Your Email].

A synthetic sample file (`sample_input.csv`) with the same column structure but randomised values is provided for testing the pipeline.

## Offshore Predictor Data (CMEMS / HYCOM)

Offshore wave parameters and wind fields were obtained from the Copernicus Marine Environment Monitoring Service (CMEMS) and the HYbrid Coordinate Ocean Model (HYCOM). These datasets are publicly available:

- CMEMS: [https://marine.copernicus.eu](https://marine.copernicus.eu)
- HYCOM: [https://www.hycom.org](https://www.hycom.org)

## Bathymetry (GEBCO)

The bathymetric data are from the GEBCO 2024 Grid, available at:

- [https://www.gebco.net/data_and_products/gridded_bathymetry_data/](https://www.gebco.net/data_and_products/gridded_bathymetry_data/)

The specific NetCDF tile used in this study covers the Gulf of Oman region.
