# Bagasse Intelligence — Individual Project Dashboard

A Streamlit application for descriptive and diagnostic analytics on a **synthetic** sugarcane–bagasse supply-chain scenario. It explores mill supply, buyer demand, quotations, conversion patterns and commercial assumptions.

## Run locally

Use Python 3.11 or newer with compatible packages:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Upload the contents of this folder to the root of the GitHub repository.
2. Keep the `data/` and `models/` folders at the same level as `app.py`.
3. Set the main file path to `app.py` and deploy.

## Folder structure

```text
bagasse-intelligence/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── Sugarcane_Bagasse_Clean_Dataset.xlsx
│   └── buyer_quote_prediction_template.csv
└── models/
    └── Sugarcane_Bagasse_Tuned_RF_Model.joblib
```

## Application modules

- **Executive Overview:** dataset KPIs and descriptive trend snapshots.
- **Data Quality & Explorer:** row/column profile, missingness, duplicates, worksheet structure, preview and CSV export.
- **Model Performance:** packaged-model diagnostic scores; not independent validation when scored on training records.
- **Hyperparameter Tuning:** compare candidate classifiers using grouped cross-validation and a chronological holdout.
- **Buyer Prediction:** batch score quotation records using the saved pipeline.
- **Commercial Prioritisation:** rank opportunities using predicted conversion probability and estimated gross margin.
- **Supply & Market Insights:** explore mill, monthly supply, buyer and transaction tables.
- **EDA & Business Drivers:** time trend, category comparisons, numeric bands, correlation analysis and written interpretation prompts.

## Data dictionary / workbook

The Excel workbook contains `READ_ME_AUDIT`, `DATA_DICTIONARY`, `Mills`, `Mill_Monthly`, `Buyers`, `Quotes` and `Transactions` worksheets. The default application dataset is the `Quotes` worksheet.

## Critical academic and analytical limitations

1. All records are synthetic. The workbook is a learning/strategy-validation scenario, not observed Indian market evidence.
2. The original generator injected approximately 1.2% conversion-label flips. The exact rows are not identified; do not claim the target labels are noise-free.
3. The model performance page scores the currently loaded labelled records. If they overlap with training data, those metrics are in-sample diagnostics and must not be reported as independent test performance.
4. Hyperparameter-tuning results use a chronological holdout; the page also warns that future-period buyers may overlap with training buyers. This is not a fully unseen-buyer test.
5. Conversion probability is not a guaranteed outcome. Commercial prioritisation is a heuristic, not recognised revenue or a financial forecast.
6. Bagasse supply figures are wet tonnes. Moisture, dry-matter yield, mill captive consumption, seasonality, storage loss, transport and fire risk require real-world validation.
7. Associations and correlations are not causal effects. Discuss them as patterns/hypotheses in the synthetic scenario.

## Individual assignment alignment

The dashboard is designed primarily to support the individual-work requirements: synthetic business scenario, data preparation, descriptive analytics, diagnostic analytics, correlation-based charts and logically explained EDA. Classification/clustering/association/regression extensions may support later group work but do not replace the individual EDA requirement.

The assignment brief states 40 marks for the individual submission while its listed components total 50 marks (10 + 10 + 30). Confirm the marking allocation with the faculty.
