# Bagasse Intelligence V2

Streamlit decision-support dashboard for the synthetic sugarcane and bagasse supply-chain project.

## Run locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy
Push the contents of this folder to the root of a GitHub repository. In Streamlit Community Cloud:
- Branch: `main`
- Main file: `app.py`

## Included
- Executive overview
- Data quality explorer
- Model performance diagnostics (explicitly labelled as potentially in-sample)
- Hyperparameter tuning with chronological holdout and grouped CV
- Batch buyer quotation prediction
- Commercial opportunity ranking
- Supply and market workbook explorer
- EDA & business drivers page
- Usage and limitations

## Important limitations
All supplied data are synthetic. Prediction probabilities are not guarantees. The Model Performance page scores the currently loaded labelled data and must not be presented as independent test performance if records overlap with training. Use an untouched chronological test period for final reporting. The application does not guarantee 95% accuracy.
