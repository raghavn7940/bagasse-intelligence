from pathlib import Path
import io, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "Sugarcane_Bagasse_Hyperparameter_Tuned.xlsx"
MODEL_PATH = APP_DIR / "models" / "Sugarcane_Bagasse_Tuned_RF_Model.joblib"

st.set_page_config(page_title="Bagasse Intelligence V2 | Command Center",
                   page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #0b1220; color: #e7edf7; }
[data-testid="stSidebar"] { background: #101a2b; border-right: 1px solid #26364d; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1600px; }
h1,h2,h3 { color: #f2f6fc; letter-spacing: -0.02em; }
p, label, .stMarkdown { color: #c8d3e2; }
div[data-testid="stMetric"] { background: linear-gradient(145deg,#142238,#101b2d);
 border: 1px solid #263a54; padding: 16px 18px; border-radius: 14px; }
div[data-testid="stMetric"] label { color: #9fb2ca; }
div[data-testid="stMetricValue"] { color: #f4f8ff; }
div[data-testid="stMetricDelta"] { color: #70d6a5; }
.stButton button, .stDownloadButton button { border-radius: 9px; font-weight: 650; }
div[data-testid="stDataFrame"] { border: 1px solid #26364d; border-radius: 10px; }
.hero { padding: 25px 28px; border: 1px solid #29435d; border-radius: 18px;
 background: radial-gradient(circle at 85% 20%,#244b45 0,#172b3b 38%,#111d30 100%); margin-bottom: 18px; }
.hero-kicker { color:#8de0b4; font-size:12px; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.hero h1 { margin:7px 0 5px; font-size:34px; }
.hero p { margin:0; color:#c2d3e3; max-width:850px; }
.small-note { color:#92a6bd; font-size:12px; }
div[data-testid="stAlert"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

def load_workbook(path=DATA_PATH):
    if not path.exists():
        return None
    try:
        return pd.ExcelFile(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def read_sheet(path, sheet):
    return pd.read_excel(path, sheet_name=sheet)

def read_uploaded_or_default(upload):
    if upload is not None:
        try:
            if upload.name.lower().endswith(".csv"):
                return pd.read_csv(upload)
            xls = pd.ExcelFile(upload)
            preferred = "Clean_Quotes" if "Clean_Quotes" in xls.sheet_names else xls.sheet_names[0]
            st.sidebar.caption(f"Reading worksheet: {preferred}")
            return pd.read_excel(upload, sheet_name=preferred)
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")
            return None
    if DATA_PATH.exists():
        return read_sheet(str(DATA_PATH), "Clean_Quotes")
    return None

def engineer_features(frame, feature_cols):
    d = frame.copy()
    # Convert original Month date to the four time features used by the packaged model.
    if "Month" in d.columns:
        dt = pd.to_datetime(d["Month"], errors="coerce")
        d["Year"] = dt.dt.year
        d["Month_Num"] = dt.dt.month
        d["Month_Sin"] = np.sin(2*np.pi*d["Month_Num"]/12)
        d["Month_Cos"] = np.cos(2*np.pi*d["Month_Num"]/12)
    for col in feature_cols:
        if col not in d.columns:
            d[col] = np.nan
    return d[feature_cols]

def metrics_for(y_true, pred, prob):
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan
    }

def make_preprocessor(X):
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric, num), ("categorical", categorical, cat)], sparse_threshold=0.0)

def build_training_frame(df):
    d = df.copy()
    if "Month" in d.columns:
        dt = pd.to_datetime(d["Month"], errors="coerce")
        d["Year"] = dt.dt.year
        d["Month_Num"] = dt.dt.month
        d["Month_Sin"] = np.sin(2*np.pi*d["Month_Num"]/12)
        d["Month_Cos"] = np.cos(2*np.pi*d["Month_Num"]/12)
    return d

def fmt_pct(v):
    return "—" if pd.isna(v) else f"{v:.1%}"

st.sidebar.markdown("## 🌾 BAGASSE INTELLIGENCE")
st.sidebar.caption("Commercial analytics • ML prediction • supply decisions")
page = st.sidebar.radio("NAVIGATION", [
    "Executive Overview", "Data Quality & Explorer", "Model Performance",
    "Hyperparameter Tuning", "Buyer Prediction", "Commercial Prioritisation",
    "Supply & Market Insights", "EDA & Business Drivers", "How to Use"
])
st.sidebar.divider()
uploaded = st.sidebar.file_uploader("Replace dataset (CSV or XLSX)", type=["csv","xlsx"])
model_upload = st.sidebar.file_uploader("Optional: upload a JOBLIB model", type=["joblib"])
df = read_uploaded_or_default(uploaded)

if df is None:
    st.error("Dataset not found. Put the supplied workbook in data/ or upload a CSV/XLSX file.")
    st.stop()

st.sidebar.caption(f"Loaded dataset: {len(df):,} rows × {len(df.columns)} columns")
if "Purchase_Converted" not in df.columns:
    st.warning("This dataset does not contain Purchase_Converted. Prediction/training pages require that target field.")

# Load packaged model or user-uploaded model.
@st.cache_resource(show_spinner=False)
def load_model_from_path(path):
    return joblib.load(path)

saved = None
try:
    if model_upload is not None:
        saved = joblib.load(model_upload)
    elif MODEL_PATH.exists():
        saved = load_model_from_path(str(MODEL_PATH))
except Exception as e:
    st.sidebar.error(f"Model could not be loaded: {e}")

def render_hero(kicker, title, subtitle):
    st.markdown(f'<div class="hero"><div class="hero-kicker">{kicker}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

if page == "Executive Overview":
    render_hero("SUGAR SUPPLY CHAIN • DECISION SUPPORT", "Bagasse Intelligence Command Center",
                "A single workspace for buyer conversion, quotation economics, mill supply and commercial decision-making.")
    total = len(df)
    buyers = df["Buyer_ID"].nunique() if "Buyer_ID" in df else np.nan
    conv = df["Purchase_Converted"].mean() if "Purchase_Converted" in df else np.nan
    quotes = df["Quote_ID"].nunique() if "Quote_ID" in df else total
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Quote records", f"{total:,}")
    c2.metric("Unique buyers", f"{buyers:,.0f}" if pd.notna(buyers) else "—")
    c3.metric("Unique quotations", f"{quotes:,}")
    c4.metric("Observed conversion", fmt_pct(conv))
    st.caption("Dataset is synthetic and intended for analytics practice. These figures are not verified market results.")
    left,right = st.columns([1.3,1])
    with left:
        if "Month" in df and "Purchase_Converted" in df:
            x = df.copy()
            x["Month"] = pd.to_datetime(x["Month"], errors="coerce")
            trend = x.dropna(subset=["Month"]).groupby(x["Month"].dt.to_period("M"))["Purchase_Converted"].mean().reset_index()
            trend["Month"] = trend["Month"].astype(str)
            fig = px.line(trend, x="Month", y="Purchase_Converted", markers=True,
                          title="Monthly quote conversion rate")
            fig.update_layout(template="plotly_dark", yaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Month or conversion target is missing; trend chart unavailable.")
    with right:
        if "Industry" in df and "Purchase_Converted" in df:
            industry = df.groupby("Industry")["Purchase_Converted"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(industry, x="Purchase_Converted", y="Industry", orientation="h",
                         title="Conversion rate by buyer industry", text_auto=".0%")
            fig.update_layout(template="plotly_dark", xaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    st.subheader("Commercial data snapshot")
    candidates = [c for c in ["Offered_Quantity_Tonnes","Buyer_Monthly_Demand_Tonnes",
                               "Offered_Price_INR_per_Tonne","Estimated_Freight_INR_per_Tonne",
                               "Estimated_Gross_Margin_INR_per_Tonne"] if c in df]
    if candidates:
        st.dataframe(df[candidates].describe().T.round(2), use_container_width=True)
    st.info("Business principle: conversion probability is not the same as commercial attractiveness. Prioritise expected contribution, quality fit, supply reliability and payment risk together.")

elif page == "Data Quality & Explorer":
    render_hero("DATA FOUNDATION", "Data Quality & Explorer", "Inspect missingness, duplicates, field types, ranges and the records that feed the models.")
    a,b,c,d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{len(df.columns):,}")
    bmiss = int(df.isna().sum().sum())
    c.metric("Missing cells", f"{bmiss:,}")
    dup = int(df.duplicated().sum())
    d.metric("Duplicate rows", f"{dup:,}")
    profile = pd.DataFrame({
        "Column":df.columns,
        "Type": [str(t) for t in df.dtypes],
        "Missing":df.isna().sum().values,
        "Missing %":(df.isna().mean().values*100).round(2),
        "Unique values":df.nunique(dropna=True).values
    }).sort_values("Missing %", ascending=False)
    st.subheader("Column profile")
    st.dataframe(profile, use_container_width=True, hide_index=True)
    selected = st.selectbox("Inspect a column", df.columns)
    st.write("Value distribution")
    counts = df[selected].astype(str).value_counts(dropna=False).head(30).rename_axis(selected).reset_index(name="Count")
    if len(counts):
        fig = px.bar(counts, x=selected, y="Count", title=f"Top values: {selected}")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Data preview")
    st.dataframe(df.head(500), use_container_width=True)
    st.download_button("Download loaded data as CSV", df.to_csv(index=False).encode("utf-8"),
                       "loaded_dataset.csv", "text/csv")

elif page == "Model Performance":
    render_hero("MODEL EVALUATION", "Model Performance", "Review the packaged model and compare it with fresh training runs on a consistent holdout.")
    if saved is None:
        st.warning("No JOBLIB model loaded. The workbook can still be used for fresh training if it contains the target.")
    else:
        st.success("Packaged JOBLIB pipeline loaded successfully.")
        st.write(f"Target: **{saved.get('target','Purchase_Converted')}** · Training cutoff: **{saved.get('training_cutoff','not recorded')}**")
        st.write(f"Features expected: **{len(saved.get('feature_columns', []))}**")
        st.code(", ".join(saved.get("feature_columns", [])), language="text")
        try:
            Xp = engineer_features(df, saved["feature_columns"])
            if saved["target"] in df:
                valid = df[saved["target"]].notna()
                yv = df.loc[valid, saved["target"]].astype(int)
                probs = saved["pipeline"].predict_proba(Xp.loc[valid])[:,1]
                preds = (probs >= 0.5).astype(int)
                m = metrics_for(yv, preds, probs)
                cols = st.columns(5)
                for col,(name,val) in zip(cols,m.items()):
                    col.metric(name, fmt_pct(val))
                cm = confusion_matrix(yv,preds,labels=[0,1])
                fig = px.imshow(cm, text_auto=True, x=["Predicted no","Predicted yes"],
                                y=["Actual no","Actual yes"], color_continuous_scale="Teal",
                                title="Confusion matrix on currently loaded labelled records")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                st.warning(
                    "IMPORTANT: These metrics are calculated on the currently loaded labelled dataset. "
                    "If these records overlap with model training, this is an in-sample diagnostic—not "
                    "independent test accuracy. For academic reporting, use an untouched chronological "
                    "holdout and disclose the split dates. Do not present this panel as validated performance."
                )
        except Exception as e:
            st.error(f"Could not score this dataset with the loaded model: {e}")

elif page == "Hyperparameter Tuning":
    render_hero("MODEL LAB", "Hyperparameter Tuning", "Train candidate classifiers, tune model complexity and evaluate on a chronological holdout. The holdout is not used for tuning.")
    if "Purchase_Converted" not in df:
        st.error("Purchase_Converted target column is required.")
        st.stop()
    work = build_training_frame(df)
    target = "Purchase_Converted"
    exclude = ["Quote_ID","Buyer_ID","Mill_ID","Data_Quality_Flag",target,"Month"]
    if "Month" not in work:
        st.error("A Month column is required for a chronological train/test split.")
        st.stop()
    work["Month"] = pd.to_datetime(work["Month"], errors="coerce")
    work = work.dropna(subset=["Month",target])
    cutoff_default = pd.Timestamp("2026-04-01")
    cutoff = st.date_input("Test period starts on", value=cutoff_default.date())
    cutoff = pd.Timestamp(cutoff)
    train = work[work["Month"] < cutoff].copy()
    test = work[work["Month"] >= cutoff].copy()
    features = [c for c in work.columns if c not in exclude]
    X_train, X_test = train[features], test[features]
    y_train, y_test = train[target].astype(int), test[target].astype(int)
    st.write(f"Training records: **{len(train):,}** · Test records: **{len(test):,}** · Features: **{len(features)}**")
    if len(train)<100 or len(test)<30:
        st.error("Not enough records in one of the periods. Change the cutoff or provide more data.")
        st.stop()
    st.warning("The test period can contain buyers seen in training. This measures future quotations for a partly known buyer base, not generalisation to entirely new buyers.")
    tune_metric = st.selectbox("Cross-validation tuning objective", ["f1","accuracy","recall","precision"], index=0)
    n_iter = st.slider("Randomized search iterations", 5, 40, 12)
    run = st.button("Run tuning and compare models", type="primary")
    if run:
        with st.spinner("Training and tuning in stable single-process mode. This may take a few minutes..."):
            pre = make_preprocessor(X_train)
            candidates = {
                "Random Forest": RandomForestClassifier(random_state=42,n_jobs=1,class_weight="balanced"),
                "Extra Trees": ExtraTreesClassifier(random_state=42,n_jobs=1,class_weight="balanced"),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }
            param_distributions = {
                "Random Forest": {
                    "model__n_estimators":[100,150,250,400],
                    "model__max_depth":[None,5,8,12,16],
                    "model__min_samples_split":[2,5,10,20],
                    "model__min_samples_leaf":[1,2,4,8],
                    "model__max_features":["sqrt",0.5,0.8]
                },
                "Extra Trees": {
                    "model__n_estimators":[100,150,250,400],
                    "model__max_depth":[None,5,8,12,16],
                    "model__min_samples_split":[2,5,10,20],
                    "model__min_samples_leaf":[1,2,4,8],
                    "model__max_features":["sqrt",0.5,0.8]
                },
                "Gradient Boosting": {
                    "model__n_estimators":[50,100,150,250],
                    "model__learning_rate":[0.02,0.05,0.08,0.1],
                    "model__max_depth":[1,2,3],
                    "model__min_samples_leaf":[2,5,10],
                    "model__subsample":[0.7,0.85,1.0]
                }
            }
            results=[]; fitted={}
            groups = train["Buyer_ID"] if "Buyer_ID" in train else pd.Series(np.arange(len(train)),index=train.index)
            n_splits = min(3, groups.nunique())
            if n_splits < 2:
                st.error("At least two distinct buyers are needed for grouped cross-validation.")
                st.stop()
            cv = GroupKFold(n_splits=n_splits)
            for name, estimator in candidates.items():
                pipe = Pipeline([("preprocessor",pre),("model",estimator)])
                search = RandomizedSearchCV(pipe,param_distributions[name],n_iter=n_iter,
                                            scoring=tune_metric,cv=cv.split(X_train,y_train,groups),
                                            random_state=42,n_jobs=1,refit=True,verbose=0)
                search.fit(X_train,y_train)
                best = search.best_estimator_
                prob = best.predict_proba(X_test)[:,1]
                pred = (prob>=0.5).astype(int)
                met = metrics_for(y_test,pred,prob)
                results.append({"Model":name,"CV best score":search.best_score_,**met,
                                "Best parameters":json.dumps(search.best_params_)})
                fitted[name] = best
            result_df = pd.DataFrame(results).sort_values("F1",ascending=False)
            st.session_state["tuning_results"] = result_df
            st.session_state["tuned_models"] = fitted
            st.session_state["tuning_features"] = features
            st.session_state["tuning_cutoff"] = str(cutoff.date())
            st.session_state["tuning_test_y"] = y_test
            st.session_state["tuning_test_X"] = X_test
    if "tuning_results" in st.session_state:
        res=st.session_state["tuning_results"]
        show=res.drop(columns=["Best parameters"]).set_index("Model")
        st.subheader("Holdout comparison")
        st.dataframe(show.style.format("{:.3f}"),use_container_width=True)
        metric=st.selectbox("Visualise metric",["Accuracy","Precision","Recall","F1","ROC-AUC"])
        fig=px.bar(res,x="Model",y=metric,color="Model",text_auto=".1%",title=f"{metric} by model")
        fig.update_layout(template="plotly_dark",showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
        chosen=st.selectbox("Select model to export",list(st.session_state["tuned_models"].keys()))
        st.json(json.loads(res.loc[res["Model"]==chosen,"Best parameters"].iloc[0]))
        export=st.button("Export selected tuned model as JOBLIB")
        if export:
            bundle={"pipeline":st.session_state["tuned_models"][chosen],
                    "feature_columns":st.session_state["tuning_features"],
                    "target":"Purchase_Converted","training_cutoff":st.session_state["tuning_cutoff"],
                    "excluded_columns":["Quote_ID","Buyer_ID","Mill_ID","Data_Quality_Flag","Purchase_Converted","Month"],
                    "note":"Retrained from the app. Check the validation report before deployment."}
            b=io.BytesIO(); joblib.dump(bundle,b)
            st.download_button("Download tuned JOBLIB model",b.getvalue(),
                               f"{chosen.lower().replace(' ','_')}_tuned.joblib",
                               "application/octet-stream")
        st.download_button("Download model comparison CSV",res.to_csv(index=False).encode(),
                           "model_comparison.csv","text/csv")

elif page == "Buyer Prediction":
    render_hero("SCORING WORKSPACE", "Buyer Prediction", "Score new quotations using the packaged pipeline. Results are probabilities, not guaranteed outcomes.")
    if saved is None:
        st.error("Load the supplied JOBLIB model from models/ or upload it in the sidebar.")
        st.stop()
    feat = saved["feature_columns"]
    st.write("Expected model fields:", ", ".join(feat))
    template = pd.DataFrame([{c: ("" if c in ["Industry","Buyer_Region","On_Time_Promise","Quality_Sensitivity","Contract_Preference"] else 0) for c in feat}])
    with st.expander("Download a prediction template"):
        st.download_button("Download CSV template",template.to_csv(index=False).encode(),
                           "buyer_quote_template.csv","text/csv")
    threshold=st.slider("Decision threshold",0.05,0.95,0.50,0.01,
                        help="Affects the Yes/No decision, not the underlying probability.")
    pred_file=st.file_uploader("Upload new quotation records (CSV or XLSX)",type=["csv","xlsx"],key="predfile")
    if pred_file:
        try:
            new = pd.read_csv(pred_file) if pred_file.name.lower().endswith(".csv") else pd.read_excel(pred_file)
            missing = [c for c in feat if c not in new.columns]
            if missing:
                st.warning("Missing model input columns will be imputed by the saved pipeline: " + ", ".join(missing))
            Xnew=engineer_features(new,feat)
            p=saved["pipeline"].predict_proba(Xnew)[:,1]
            out=new.copy()
            out["Conversion_Probability"]=p
            out["Predicted_Conversion"]=(p>=threshold).astype(int)
            out["Priority"]=pd.cut(p,[-0.001,0.35,0.65,1.001],labels=["Low","Medium","High"])
            if "Estimated_Gross_Margin_INR_per_Tonne" in out.columns and "Offered_Quantity_Tonnes" in out.columns:
                q = pd.to_numeric(out["Offered_Quantity_Tonnes"],errors="coerce").fillna(0)
                m = pd.to_numeric(out["Estimated_Gross_Margin_INR_per_Tonne"],errors="coerce").fillna(0)
                out["Estimated_Opportunity_Margin_INR"] = q*m
                out["Expected_Weighted_Margin_INR"] = p*out["Estimated_Opportunity_Margin_INR"]
            a,b,c,d = st.columns(4)
            a.metric("Quotations scored", f"{len(out):,}")
            b.metric("Predicted conversions", int(out["Predicted_Conversion"].sum()))
            c.metric("High-priority buyers", int((out["Priority"]=="High").sum()))
            d.metric("Mean conversion probability", f"{p.mean():.1%}" if len(p) else "—")
            st.caption("Probability is a model estimate. Threshold changes the binary decision only; it does not improve model quality.")
            st.dataframe(out.sort_values("Conversion_Probability",ascending=False),use_container_width=True)
            st.download_button("Download scored quotations",out.to_csv(index=False).encode(),
                               "scored_buyer_quotes.csv","text/csv")
        except Exception as e:
            st.error(f"Prediction failed. Check the uploaded file and model compatibility. Details: {e}")

elif page == "Commercial Prioritisation":
    render_hero("PROFIT-LED SELLING", "Commercial Prioritisation", "Rank opportunities using conversion probability and estimated per-tonne gross margin. Confirm costs before acting.")
    if saved is None:
        st.error("A JOBLIB model is required for conversion scoring.")
        st.stop()
    if "Estimated_Gross_Margin_INR_per_Tonne" not in df:
        st.error("Estimated_Gross_Margin_INR_per_Tonne is missing.")
        st.stop()
    Xp=engineer_features(df,saved["feature_columns"])
    try:
        prob=saved["pipeline"].predict_proba(Xp)[:,1]
        ranked=df.copy()
        ranked["Conversion_Probability"]=prob
        if "Offered_Quantity_Tonnes" not in ranked.columns:
            st.error("Offered_Quantity_Tonnes is missing; opportunity value cannot be calculated.")
            st.stop()
        ranked["Estimated_Opportunity_Margin_INR"]=(
            pd.to_numeric(ranked["Offered_Quantity_Tonnes"],errors="coerce").fillna(0) *
            pd.to_numeric(ranked["Estimated_Gross_Margin_INR_per_Tonne"],errors="coerce").fillna(0)
        )
        ranked["Expected_Weighted_Margin_INR"]=ranked["Conversion_Probability"]*ranked["Estimated_Opportunity_Margin_INR"]
        ranked=ranked.sort_values("Expected_Weighted_Margin_INR",ascending=False)
        c1,c2,c3=st.columns(3)
        c1.metric("Expected weighted margin (all quotes)",f"₹{ranked['Expected_Weighted_Margin_INR'].sum():,.0f}")
        c2.metric("Positive-margin quotes",int((ranked["Estimated_Opportunity_Margin_INR"]>0).sum()))
        c3.metric("High probability + positive margin",int(((ranked["Conversion_Probability"]>=.65)&(ranked["Estimated_Opportunity_Margin_INR"]>0)).sum()))
        fields=[c for c in ["Quote_ID","Buyer_ID","Industry","Offered_Quantity_Tonnes",
                "Conversion_Probability","Estimated_Gross_Margin_INR_per_Tonne",
                "Estimated_Opportunity_Margin_INR","Expected_Weighted_Margin_INR"] if c in ranked]
        st.dataframe(ranked[fields].head(300),use_container_width=True)
        st.download_button("Download ranked commercial opportunities",ranked.to_csv(index=False).encode(),
                           "commercially_ranked_opportunities.csv","text/csv")
        st.caption("Expected weighted margin = modelled conversion probability × estimated gross opportunity margin. It is a prioritisation heuristic, not recognised revenue or a forecast.")
    except Exception as e:
        st.error(f"Could not calculate prioritisation: {e}")

elif page == "Supply & Market Insights":
    render_hero("SUPPLY CHAIN LENS", "Supply & Market Insights", "Explore the workbook's mill, monthly supply, buyer segment, transaction and association-rule tabs.")
    if not DATA_PATH.exists():
        st.info("Supply insights require the supplied Excel workbook in the data folder.")
        st.stop()
    book=pd.ExcelFile(DATA_PATH)
    tabs=[t for t in ["Mills","Mill_Monthly","Buyers_Segmented","Transactions","Cluster_Profiles","Association_Rules","Regression_Metrics"] if t in book.sheet_names]
    tab=st.selectbox("Choose a workbook analysis",tabs)
    sheet=pd.read_excel(DATA_PATH,sheet_name=tab)
    st.subheader(tab.replace("_"," "))
    st.dataframe(sheet,use_container_width=True)
    numeric=sheet.select_dtypes(include=np.number).columns.tolist()
    if numeric:
        col=st.selectbox("Plot numeric field",numeric)
        if len(sheet)>0:
            fig=px.histogram(sheet,x=col,title=f"Distribution of {col}")
            fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
    st.caption("These workbook analyses are synthetic and exploratory. Validate actual mill surplus, captive use, moisture and buyer offtake before making supply commitments.")


elif page == "EDA & Business Drivers":
    render_hero("EXPLORATORY ANALYTICS", "EDA & Business Drivers",
                "Explore how quotation, buyer, quality and logistics variables relate to conversion. Associations are descriptive, not causal.")
    if "Purchase_Converted" not in df:
        st.error("Purchase_Converted is required for conversion analysis.")
        st.stop()
    target = pd.to_numeric(df["Purchase_Converted"], errors="coerce")
    valid = df.loc[target.notna()].copy()
    valid["Purchase_Converted"] = target.dropna().astype(int)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows analysed", f"{len(valid):,}")
    c2.metric("Conversion rate", f"{valid['Purchase_Converted'].mean():.1%}")
    c3.metric("Non-conversion rate", f"{1-valid['Purchase_Converted'].mean():.1%}")

    cat_candidates = [c for c in ["Industry","Buyer_Region","Payment_Terms","Contract_Preference",
                                  "Quality_Sensitivity","On_Time_Promise"] if c in valid.columns]
    num_candidates = [c for c in ["Offered_Price_INR_per_Tonne","Estimated_Freight_INR_per_Tonne",
                                  "Offered_Moisture_Pct","Contamination_Pct","Delivery_Distance_km",
                                  "Buyer_Monthly_Demand_Tonnes","Offered_Quantity_Tonnes",
                                  "Estimated_Gross_Margin_INR_per_Tonne"] if c in valid.columns]
    left, right = st.columns(2)
    with left:
        if cat_candidates:
            cat = st.selectbox("Compare conversion by category", cat_candidates)
            agg = valid.groupby(cat, dropna=False)["Purchase_Converted"].agg(["mean","count"]).reset_index()
            agg = agg.rename(columns={"mean":"Conversion_Rate","count":"Quotes"})
            fig = px.bar(agg, x=cat, y="Conversion_Rate", text="Quotes",
                         title=f"Conversion rate by {cat}", hover_data=["Quotes"])
            fig.update_layout(template="plotly_dark", yaxis_tickformat=".0%",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(agg.sort_values("Conversion_Rate",ascending=False),use_container_width=True,hide_index=True)
    with right:
        if num_candidates:
            num = st.selectbox("Compare conversion across numeric variable", num_candidates)
            temp = valid[[num,"Purchase_Converted"]].copy()
            temp[num] = pd.to_numeric(temp[num], errors="coerce")
            temp = temp.dropna()
            if len(temp):
                temp["Value_Band"] = pd.qcut(temp[num], q=min(5,temp[num].nunique()),
                                             duplicates="drop")
                agg = temp.groupby("Value_Band",observed=True)["Purchase_Converted"].agg(["mean","count"]).reset_index()
                agg["Value_Band"] = agg["Value_Band"].astype(str)
                agg = agg.rename(columns={"mean":"Conversion_Rate","count":"Quotes"})
                fig = px.line(agg, x="Value_Band", y="Conversion_Rate", markers=True,
                              title=f"Conversion by {num} band", hover_data=["Quotes"])
                fig.update_layout(template="plotly_dark", yaxis_tickformat=".0%",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(agg,use_container_width=True,hide_index=True)
    numeric_df = valid.select_dtypes(include=np.number)
    if numeric_df.shape[1] >= 2:
        st.subheader("Correlation matrix")
        corr = numeric_df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu",
                        zmin=-1, zmax=1, title="Pearson correlation among numeric fields")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Correlation measures linear association. It does not establish causation; ID-like and derived fields should be excluded before drawing substantive conclusions.")
    st.info("Use these charts to generate hypotheses for further testing. Differences across categories may reflect confounding variables such as price, region, buyer history or supply conditions.")

elif page == "How to Use":
    render_hero("GETTING STARTED", "How to Use This Application", "The project folder is self-contained. Install dependencies once, then launch the app locally.")
    st.markdown("""
### 1. Install Python
Use Python 3.10 or 3.11. Open Terminal / Command Prompt inside this folder.

### 2. Install the libraries
```bash
pip install -r requirements.txt
```

### 3. Start the application
```bash
streamlit run app.py
```
The app opens in your browser, normally at `http://localhost:8501`.

### 4. Folder structure
- `app.py` — Streamlit application.
- `data/` — Excel workbook.
- `models/` — trained JOBLIB pipeline.
- `requirements.txt` — Python dependencies.
- `README.md` — setup and operating instructions.

### 5. Important model limitations
- The supplied dataset is synthetic, not actual market evidence.
- The packaged model's reported 60.1% chronological holdout accuracy is modest.
- Do not use in-sample metrics as evidence of future accuracy.
- The model predicts quote conversion, not credit approval, quality compliance or profitability.
- The opportunity-margin ranking is only as reliable as the cost and margin inputs.
- For a stronger test, hold out later months and separately test completely unseen buyers.
""")
    st.info("Before real commercial deployment, replace synthetic data with verified mill, buyer, logistics, quality and transaction records.")
