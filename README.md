# Drift-Aware Selective Prediction for Reliable Transformer Classification Under Distribution Shift

> Research project focused on drift-aware selective prediction and transformer reliability under semantic distribution shift using Mahalanobis embedding-space drift estimation. 

This repository contains research work investigating reliability-aware selective prediction for transformer-based text classification models operating under semantic distribution shift. The project introduces **Drift-Aware Selective Calibration (DASC)**, a lightweight inference-time framework that combines semantic drift estimation with adaptive confidence modulation to selectively abstain from unreliable predictions under out-of-domain conditions. 

## Overview

Transformer models frequently remain overconfident under semantic distribution shift, producing highly confident yet unreliable predictions in out-of-domain environments. Conventional confidence calibration approaches improve probability estimation but still force predictions for semantically shifted inputs.

This project investigates how representation-space semantic drift can be integrated directly into selective prediction mechanisms for improving transformer reliability under distribution shift. The proposed DASC framework combines Mahalanobis embedding-space drift estimation with adaptive confidence modulation to selectively reduce operational trust for semantically unreliable samples. 

## Objectives

* Investigate transformer reliability under semantic distribution shift
* Analyze embedding-space semantic drift using Mahalanobis distance
* Evaluate confidence overconfidence under out-of-domain conditions
* Develop drift-aware selective prediction mechanisms
* Improve operational reliability through adaptive abstention
* Study reliability–coverage tradeoffs in selective prediction systems

## Methods and Approaches

The project utilizes:

* Transformer-based text classification
* DistilBERT semantic embeddings
* Mahalanobis drift estimation
* Drift-aware confidence modulation
* Selective prediction and abstention
* Risk–coverage analysis
* Out-of-domain reliability evaluation
* Embedding-space uncertainty analysis

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Datasets
* Scikit-learn
* NumPy
* Pandas
* SciPy
* Matplotlib
* Jupyter Notebook

## Repository Structure

```text
drift-aware-selective-calibration/
│
├── notebooks/
│   └── drift_aware_selective_calibration.ipynb
│
├── figures/
│   ├── mahalanobis_drift_distribution.png
│   ├── confidence_vs_semantic_drift.png
│   ├── risk_coverage_curve_dasc.png
│   ├── alpha_sensitivity_selective_accuracy.png
│   ├── alpha_sensitivity_coverage.png
│   ├── alpha_sensitivity_aurc.png
│   ├── in_domain_vs_ood_drift_distribution.png
│   ├── confidence_under_distribution_shift.png
│   ├── coverage_under_semantic_shift.png
│   └── dasc_confidence_distribution_shift.png
│
├── paper/
│   └── DASC_Sharma.pdf
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Drift-Aware Selective Calibration (DASC)

The proposed framework integrates predictive confidence with semantic drift estimation using:

```math
S(x) = c(x)\exp(-\alpha \tilde{D}(x))
```

where:

* `c(x)` represents predictive confidence,
* `\tilde{D}(x)` denotes normalized semantic drift,
* and `\alpha` controls drift sensitivity. 

Unlike conventional confidence thresholding, DASC conditions prediction reliability on both:

* predictive confidence,
* and embedding-space semantic deviation.

This enables adaptive abstention for semantically unreliable predictions under distribution shift without requiring retraining or architectural modification. 

## Experimental Setup

### In-Domain Dataset

* AG News

### Out-of-Domain Dataset

* Yahoo Answers Topics

### Transformer Backbone

* DistilBERT (`distilbert-base-uncased`)

### Core Experimental Components

* CLS embedding extraction
* Mahalanobis drift estimation
* Drift-aware confidence modulation
* Selective prediction under semantic shift
* Risk–coverage evaluation
* AURC analysis
* Drift sensitivity evaluation

## Evaluation Metrics

The project evaluates reliability and selective prediction performance using:

* Classification Accuracy
* Macro-F1 Score
* Predictive Confidence
* Selective Accuracy
* Selective Risk
* Coverage
* Risk–Coverage Curves
* Area Under the Risk–Coverage Curve (AURC)
* Semantic Drift Analysis

## Key Findings

* Transformer confidence remains substantially overconfident under semantic distribution shift.
* Mahalanobis embedding-space drift effectively captures semantic deviation from the training distribution.
* Confidence-only selective prediction is insufficient under semantic drift.
* Drift-aware selective prediction improves retained-sample reliability through adaptive abstention.
* DASC enables controllable reliability–coverage tradeoffs governed by drift sensitivity.
* Representation-space semantic drift provides reliability information beyond predictive confidence alone. 

## Figures and Visualizations

The repository includes:

* Mahalanobis drift distribution analysis
* Confidence versus semantic drift visualization
* Risk–coverage comparison curves
* Alpha sensitivity analysis
* In-domain versus out-of-domain drift distributions
* Confidence behavior under distribution shift
* Drift-aware confidence distribution analysis
* Coverage reduction under semantic drift

## Research Contributions

This work introduces:

* Drift-Aware Selective Calibration (DASC)
* Representation-aware selective prediction
* Drift-aware transformer abstention
* Lightweight inference-time reliability control
* Embedding-space semantic reliability analysis

The framework operates entirely at inference time without:

* retraining,
* ensemble methods,
* Bayesian inference,
* or architectural modification. 

## Research Paper

The accompanying manuscript:

**“Drift-Aware Selective Prediction for Reliable Transformer Classification Under Distribution Shift”**

is included in the `paper/` directory for academic reference and reproducibility purposes. 

## Reproducibility

### Installation

```bash
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/drift_aware_selective_calibration.ipynb
```

## Future Work

* Evaluation across larger transformer architectures
* Adaptive drift-aware calibration mechanisms
* Integration with conformal prediction methods
* Bayesian uncertainty estimation under semantic shift
* Reliability analysis for generative language models
* Multimodal selective prediction systems
* Theoretical analysis of semantic drift and selective risk
* Trustworthy AI systems under dynamic distribution shift

## License

This project is licensed under the MIT License.

## Author

Shivaansh Sharma
