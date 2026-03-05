# Gaussian Mixture Model — Bayesian Inference with NUTS

A Bayesian Gaussian Mixture Model implementation using NUTS sampling, with a PyMC script and equivalent Stan model.

## Project Structure

```
mixture_modeling/
├── data/
├── models/
│   └── gaussian_mixture.stan
├── outputs/
│   └── gmm_results.png
├── scripts/
│   └── gmm_pymc.py
├── requirements.txt
└── README.md
```

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the model**
```bash
python scripts/gmm_pymc.py
```

Results will be saved to `outputs/gmm_results.png`.
