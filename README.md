# 🍊 Predicting Orange Quality with Machine Learning + SHAP

Predict an orange's 1–5 quality score from its measurable properties, explain every
prediction with SHAP, and measure how much accuracy is lost when only
**non-destructive** measurements are available.

> **Part 2 of 5** in my portfolio on data science and deep learning for non-destructive
> fruit quality detection.
> ← [Part 1: Exploratory analysis](https://github.com/<your-username>/orange-quality-analysis)
> · Part 3: Brix prediction from NIR spectra →

---

## Research questions

1. How accurately can we predict quality compared with a naive baseline?
2. Which algorithm works best on a small tabular dataset?
3. Which features drive the predictions, globally and for individual oranges?
4. **Sensor ladder:** how much accuracy do we lose using only properties that can be
   measured without cutting the fruit?

## Approach

| Step | What I did | Why |
|---|---|---|
| Split | 80/20 train/test, stratified by quality class | Test set touched **once**, at the end |
| Pipeline | Scaling + one-hot encoding inside a scikit-learn `Pipeline` | No data leakage into CV folds |
| Model comparison | Baseline, Ridge, Random Forest, Gradient Boosting, XGBoost | 5-fold CV repeated 3× → mean ± SD |
| Tuning | `RandomizedSearchCV` (40 configs) for XGBoost | Same repeated CV |
| Explainability | SHAP (global bar, beeswarm, dependence, waterfall) + permutation importance | Two independent methods to cross-check |
| Sensor ladder | Same model on 3 feature sets: camera+scale → +firmness/storage → +Brix/pH | Which sensors a grading line actually needs |

## Results

<!-- TODO: fill in from notebooks/02_modeling.ipynb (section 8 prints a summary) -->

**Model comparison (cross-validation)**

| Model | CV RMSE | CV R² |
|---|---|---|
| Baseline (mean) | … | … |
| Ridge regression | … | … |
| Random Forest | … | … |
| Gradient Boosting | … | … |
| XGBoost (tuned) | … | … |

**Final model on the held-out test set:** RMSE … · MAE … · R² … ·
…% of oranges predicted within ±0.5 points.

![Model comparison](reports/figures/01_model_comparison.png)
![Predicted vs actual](reports/figures/02_test_predictions.png)

### What drives quality? (SHAP)

![SHAP importance](reports/figures/03_shap_importance.png)
![SHAP beeswarm](reports/figures/04_shap_beeswarm.png)

<!-- TODO: 2–3 sentences: top features, direction of effect, do SHAP and permutation importance agree? -->

### Sensor ladder

![Sensor ladder](reports/figures/08_sensor_ladder.png)

<!-- TODO: e.g. "Camera and scale alone explain only …% of the variance; adding Brix and pH
raises it to …%. A non-destructive grading line therefore needs a way to estimate sugar
content — which is exactly what NIR spectroscopy offers (Project 3)." -->

## Interactive predictor

Set the properties of an orange in the sidebar → get a predicted quality score and a
SHAP waterfall chart explaining it.

```bash
python -m src.train     # trains and saves models/quality_model.joblib
streamlit run app.py
```

<!-- TODO: add a screenshot: reports/figures/app.png -->

## Project structure

```
orange-quality-ml/
├── app.py                    # Streamlit predictor with SHAP explanations
├── data/README.md            # where to put orange_quality.csv
├── notebooks/
│   └── 02_modeling.ipynb     # full workflow with commentary
├── src/
│   ├── data_utils.py         # loading, cleaning, feature engineering (from Project 1)
│   ├── modeling.py           # feature sets, pipelines, CV, metrics, SHAP helpers
│   └── train.py              # reproducible training script
├── models/                   # saved model (created by train.py / notebook)
├── reports/
│   ├── figures/
│   └── metrics.json
└── requirements.txt
```

## How to run

```bash
git clone https://github.com/<your-username>/orange-quality-ml.git
cd orange-quality-ml
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put the Kaggle CSV at data/orange_quality.csv
jupyter notebook notebooks/02_modeling.ipynb
```

## Limitations

- Small dataset → wide error bars; the test set has only ~50 oranges.
- SHAP explains the model, not the biology. Correlated features (Brix and Brix/pH) share credit.
- Brix and pH were measured destructively; in practice they would be predicted from
  spectra, adding error that is not reflected here.

## Author

**<Your Name>**: Data Science & Machine Learning
[LinkedIn](https://linkedin.com/in/<you>) · [Email](mailto:<you>@example.com)
