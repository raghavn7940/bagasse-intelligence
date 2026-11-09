# Sugarcane & Bagasse AI Command Center

A local Streamlit application for exploring the synthetic sugarcane/bagasse dataset, reviewing model performance, tuning classification models, scoring new buyer quotations, and ranking commercial opportunities.

## Included files

- `app.py` — main Streamlit app
- `data/Sugarcane_Bagasse_Hyperparameter_Tuned.xlsx` — dataset and analysis workbook
- `models/Sugarcane_Bagasse_Tuned_RF_Model.joblib` — trained Random Forest pipeline
- `requirements.txt` — Python dependencies

## Run locally

1. Install Python 3.10 or 3.11.
2. Extract this folder.
3. Open Terminal / Command Prompt in this folder.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

6. Open the local address shown in the terminal (normally `http://localhost:8501`).

## Main sections

1. Executive Overview
2. Data Quality & Explorer
3. Model Performance
4. Hyperparameter Tuning
5. Buyer Prediction
6. Commercial Prioritisation
7. Supply & Market Insights
8. How to Use

## Important model notes

- All supplied data is synthetic and is for learning/prototyping, not evidence of real buyer or mill behaviour.
- The bundled model predicts `Purchase_Converted`.
- It expects the feature fields listed in the model bundle. If the uploaded file contains a `Month` date field, the app derives `Year`, `Month_Num`, `Month_Sin` and `Month_Cos`.
- The reported tuned Random Forest performance in the workbook is based on a chronological holdout and is modest. Do not treat it as production-ready.
- The packaged model's evaluation on the full loaded dataset is an in-sample diagnostic and must not be reported as unbiased test performance.
- Hyperparameter tuning uses grouped cross-validation by buyer on the training period and evaluates the untouched later-period holdout.
- The model does not determine whether a buyer is creditworthy, whether bagasse meets quality specifications, or whether supply is available.
- Commercial prioritisation uses probability × estimated opportunity margin as a heuristic, not booked revenue or a financial forecast.

## Before commercial use

Replace synthetic records with verified, timestamped data. Capture buyer-level history, quote response times, competitor pricing, actual freight, measured moisture and contamination, mill captive use, genuine surplus availability, payment/default history, and realised contribution margin. Validate on later time periods and unseen buyers before deployment.
