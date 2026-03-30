# Amazon Reviews — Sentiment Analysis Pipeline
**SSIM916 Problem Set #2**

This repository contains a four-notebook NLP pipeline that applies sentiment analysis to Amazon Pet Supplies reviews to identify critical complaints and product-specific negative keywords.

---

## Repository Structure

```
├── 01_data_cleaning.ipynb        # Load, clean, and sample the raw dataset
├── 02_tfidf_logreg.ipynb         # TF-IDF + Logistic Regression baseline model
├── 03_roberta.ipynb              # RoBERTa transfer learning model
├── 04_business_insights.ipynb    # Complaint urgency and keyword analysis
└── data/
    └── cleaned_reviews.csv       # Pre-cleaned dataset (10,000 rows) for replication
```

---

## Requirements

Install all dependencies before running any notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn transformers torch tqdm joblib scipy
```

Python 3.10 or above is recommended.

---

## How to Run the Pipeline

Notebooks must be run **in order**. Each notebook reads from the output of the previous one.

```
01_data_cleaning.ipynb
        ↓
02_tfidf_logreg.ipynb
        ↓
03_roberta.ipynb
        ↓
04_business_insights.ipynb
```

> **Note:** Notebook 3 (RoBERTa) runs inference on 10,000 reviews using CPU by default. This may take 20–40 minutes depending on your machine. If a GPU is available, inference time will be significantly reduced.

---

## Dataset

### Option A — Use the Pre-cleaned Dataset (Recommended for Replication)

The `data/` folder contains `cleaned_reviews.csv`, a pre-cleaned and sampled file of **10,000 reviews** ready for use. This file was produced by Notebook 1 and is provided to avoid the need to download the full 8.35 GB raw dataset.

To use this option, **skip Notebook 1** and begin directly from Notebook 2. Ensure `cleaned_reviews.csv` remains inside the `data/` folder.

---

### Option B — Download the Raw Dataset and Run from Notebook 1

The raw dataset is sourced from the McAuley-Lab/Amazon-Reviews-2023 collection on HuggingFace.

**Direct link to the raw file:**
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/review_categories

Download the file named `Pet_Supplies.jsonl` and place it in the same directory as the notebooks.

**Alternatively, download via Python:**

```python
from datasets import load_dataset
import json

dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Pet_Supplies",
    trust_remote_code=True
)

with open("Pet_Supplies.jsonl", "w") as f:
    for record in dataset["full"]:
        f.write(json.dumps(record) + "\n")
```

> **Warning:** The full raw file is approximately 8.35 GB. Ensure sufficient disk space and a stable internet connection before downloading.

Once downloaded, open `01_data_cleaning.ipynb` and update the `DATA_PATH` variable in the configuration cell to point to your downloaded file:

```python
DATA_PATH = Path("Pet_Supplies.jsonl")   # update this line
```

Then set `SAMPLE_SIZE` to your desired output row count:

```python
SAMPLE_SIZE = 10_000   # recommended for replication
```

Run all cells in Notebook 1 before proceeding to Notebook 2.

---

## Reproducibility

A fixed random seed of `42` is applied throughout the pipeline. As long as the same `SAMPLE_SIZE` and input file are used, all results will be identical across runs. The `cleaned_reviews.csv` in the `data/` folder was produced with `SAMPLE_SIZE = 10_000` and `random_state = 42`.

---

## Expected Outputs

All figures and data files are saved automatically to the `data/` folder during each notebook run.

| File | Produced by | Description |
|---|---|---|
| `data/cleaned_reviews.csv` | Notebook 1 | Cleaned and sampled dataset with sentiment labels |
| `data/fig_rating_distribution.png` | Notebook 1 | Rating and sentiment distribution chart |
| `data/fig_review_length.png` | Notebook 1 | Review length distribution chart |
| `data/fig_cm_tfidf.png` | Notebook 2 | Confusion matrix for TF-IDF + Logistic Regression |
| `data/fig_tfidf_terms.png` | Notebook 2 | Top predictive terms per sentiment class |
| `models/tfidf_logreg.pkl` | Notebook 2 | Saved TF-IDF + Logistic Regression model |
| `data/cleaned_reviews_with_preds.csv` | Notebooks 2 & 3 | Dataset with predictions from both models |
| `data/model_comparison.csv` | Notebook 3 | Side-by-side performance comparison table |
| `data/fig_cm_roberta.png` | Notebook 3 | Confusion matrix for RoBERTa |
| `data/fig_roberta_confidence.png` | Notebook 3 | RoBERTa prediction confidence distribution |
| `data/fig_priority_tiers.png` | Notebook 4 | Complaint urgency priority tier chart |
| `data/fig_neg_keywords.png` | Notebook 4 | Top negative keyword frequency chart |
| `data/fig_keyword_excess.png` | Notebook 4 | Complaint-specific keyword excess rate chart |
| `data/brand_response_framework.csv` | Notebook 4 | Three-tier brand response framework |
| `data/critical_reviews_sample.csv` | Notebook 4 | Top 10 critical reviews by helpful vote count |
| `data/negative_keywords.csv` | Notebook 4 | Full negative keyword frequency and excess rate table |
