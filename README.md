# A Hybrid Intelligence Framework for Computationally Efficient and Trustworthy Coastal Wave Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official code and data for the research paper, "A Hybrid Intelligence Framework for Computationally Efficient and Trustworthy Coastal Wave Forecasting," submitted to *Expert Systems with Applications*.

## Abstract

Accurate and timely forecasting of coastal wave conditions is critical for maritime safety and operations. This work introduces a novel hybrid intelligence framework that synergizes causal inference, targeted feature engineering, and state-of-the-art machine learning to create a computationally efficient, highly accurate, and transparent coastal wave forecasting system. Our results demonstrate that this decoupled approach provides a robust and trustworthy alternative to traditional modeling paradigms for complex geophysical systems.

![Framework Diagram](figures/Fig_1_Framework_Diagram.png)
*Fig 1. The three-phase Hybrid Intelligence Framework.*

## Key Features

-   **Causal Predictor Selection:** Utilizes the PCMCI+ algorithm to identify a parsimonious and physically relevant subset of offshore predictors.
-   **Comparative Modeling:** Provides a rigorous benchmark of four models (XGBoost, LightGBM, Random Forest, EBM) using a Nested, Blocked Cross-Validation scheme.
-   **Trustworthy AI:** Demonstrates a "convergent evidence" paradigm where multiple models independently agree on the key physical drivers of wave events.
-   **Decision Support System (DSS):** Implements a complete DSS with probabilistic forecasting (Conformal Prediction), automated recommendations (VIKOR), and explainability (XAI).

## Repository Structure

-   `/notebooks`: Contains the Jupyter notebooks for each stage of the project, from feature engineering to the final DSS implementation.
-   `/data`: Contains the final, engineered dataset used for model training and evaluation.
-   `/figures`: Contains all the final, publication-quality figures presented in the manuscript.
-   `/src`: (Optional) Contains reusable Python utility functions.

## Installation & Setup

To replicate the results, please follow these steps:

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/Hybrid-Wave-Forecasting-DSS.git](https://github.com/YourUsername/Hybrid-Wave-Forecasting-DSS.git)
    cd Hybrid-Wave-Forecasting-DSS
    ```

2.  Create a Python virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The cuML library for the GPU-accelerated Random Forest requires a specific CUDA-enabled environment. Please refer to the official [RAPIDS installation guide](https://rapids.ai/start.html) for setup.*

## Usage

The Jupyter notebooks in the `/notebooks` directory are numbered in the order they should be run:

1.  `1_Feature_Engineering.ipynb`: Loads the raw data and generates the final feature set.
2.  `2_EBM_Model.ipynb` to `5_Random_Forest_Model.ipynb`: Train and evaluate each of the four models.
3.  `6_DSS_Implementation.ipynb`: Implements the final Decision Support System using the champion model.

## Citation

If you use this code or our findings in your research, please cite our paper:

```bibtex
@article{Djawadi2025,
  title={A Hybrid Intelligence Framework for Computationally Efficient and Trustworthy Coastal Wave Forecasting},
  author={Djawadi, M. H. S. and [Your Supervisor's Name]},
  journal={Expert Systems with Applications},
  year={2025},
  publisher={Elsevier}
}