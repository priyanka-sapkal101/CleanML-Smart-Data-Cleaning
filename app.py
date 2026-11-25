import io
import hashlib
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CleanML - Smart Data Cleaning",
    page_icon="🧹",
    layout="wide"
)


# =========================================================
# 2. SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "dataset_key": None,
    "original_data": None,
    "cleaned_data": None,
    "cleaning_history": [],
    "cleaning_summary": {},
    "cleaning_mode": None,
    "last_clean_settings": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_dataset_state(dataset_key, dataframe):
    """Reset all cleaning results when a different file is uploaded."""
    st.session_state["dataset_key"] = dataset_key
    st.session_state["original_data"] = dataframe.copy()
    st.session_state["cleaned_data"] = None
    st.session_state["cleaning_history"] = []
    st.session_state["cleaning_summary"] = {}
    st.session_state["cleaning_mode"] = None
    st.session_state["last_clean_settings"] = None
    st.session_state["cleaning_history"] = []


def log_cleaning_actions(mode, actions):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for action in actions:
        st.session_state["cleaning_history"].append({
            "Time": timestamp,
            "Action": mode,
            "Details": action
        })


# =========================================================
# 3. ALGORITHM AND COLUMN HELPERS
# =========================================================

MODEL_OPTIONS = [
    "Linear Regression",
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors (KNN)",
    "Support Vector Machine (SVM)",
    "Naive Bayes"
]

SCALE_MODELS = {
    "Linear Regression",
    "Logistic Regression",
    "K-Nearest Neighbors (KNN)",
    "Support Vector Machine (SVM)",
    "Naive Bayes"
}


def get_problem_type(model_name, target_series):
    if model_name == "Linear Regression":
        return "Regression"
    if model_name in {"Logistic Regression", "Naive Bayes"}:
        return "Classification"

    non_null = target_series.dropna()
    unique_count = non_null.nunique()

    if pd.api.types.is_numeric_dtype(non_null) and unique_count > 15:
        return "Regression"
    return "Classification"


def safe_mode(series):
    modes = series.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else "Unknown"


def is_identifier_column(column_name, series):
    """
    Identify likely ID columns using common names and near-unique values.
    This is a heuristic; users should review the detected columns.
    """
    name = str(column_name).strip().lower().replace(" ", "_")
    common_ids = {
        "id", "index", "uuid", "guid", "customer_id", "user_id",
        "record_id", "row_id", "transaction_id", "employee_id",
        "product_id", "order_id"
    }

    if (
        name in common_ids
        or name.endswith("_id")
        or name.startswith("id_")
        or name.endswith("_uuid")
    ):
        return True

    values = series.dropna()
    if len(values) >= 20 and values.nunique() / len(values) >= 0.98:
        return True

    return False


def is_categorical(series):
    return (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    )


def detect_high_cardinality_columns(dataframe, columns, threshold):
    high_cardinality = []
    for col in columns:
        if col not in dataframe.columns:
            continue
        if is_categorical(dataframe[col]):
            unique_count = dataframe[col].nunique(dropna=True)
            if unique_count > threshold:
                high_cardinality.append(col)
    return high_cardinality


# =========================================================
# 4. DATA CLEANING ENGINE
# =========================================================

def clean_dataset(
    source_df,
    target_column,
    model_name,
    missing_numeric="Median",
    missing_categorical="Mode",
    outlier_method="Cap using IQR",
    encoding_method="One-Hot Encoding",
    scale_features=None,
    remove_duplicates=True,
    high_cardinality_method="Keep unchanged",
    high_cardinality_threshold=10,
):
    """
    Clean a copy of the uploaded data.

    Target protection:
    - Target is never imputed, encoded, capped, or scaled.
    - Rows with missing target values are removed because they cannot
      be used as labelled examples in supervised learning.

    Scaling:
    - Only original numeric input features are scaled.
    - One-Hot Encoded columns are deliberately NOT scaled.
    - Likely ID columns are not scaled or outlier-treated.
    """
    if target_column not in source_df.columns:
        raise ValueError("The selected target column does not exist.")

    result = source_df.copy()
    actions = []
    outlier_report = []
    original_rows = len(result)
    original_columns = len(result.columns)

    # Replace +/- infinity in feature columns with missing values.
    # The target column is deliberately excluded.
    original_feature_columns = [
        col for col in result.columns if col != target_column
    ]
    for col in original_feature_columns:
        if pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)

    # Remove rows without a target label.
    missing_target = int(result[target_column].isna().sum())
    if missing_target:
        result = result.dropna(subset=[target_column]).copy()
        actions.append(
            f"Removed {missing_target} rows with missing target values in '{target_column}'."
        )

    if result.empty:
        raise ValueError(
            "No rows remain after removing records with missing target values."
        )

    # Remove exact duplicate records.
    if remove_duplicates:
        duplicate_count = int(result.duplicated().sum())
        if duplicate_count:
            result = result.drop_duplicates().copy()
        actions.append(f"Removed {duplicate_count} duplicate rows.")
    else:
        actions.append("Duplicate removal was disabled.")

    feature_columns = [col for col in result.columns if col != target_column]

    # Identify the original numeric feature columns BEFORE encoding.
    # This list is used later so dummy columns are never scaled.
    original_numeric_features = [
        col for col in feature_columns
        if pd.api.types.is_numeric_dtype(result[col])
        and not is_identifier_column(col, result[col])
    ]

    # Missing-value treatment for feature columns only.
    for col in feature_columns:
        missing_count = int(result[col].isna().sum())
        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(result[col]):
            if missing_numeric == "Mean":
                fill_value = result[col].mean()
            elif missing_numeric == "Zero":
                fill_value = 0
            else:
                fill_value = result[col].median()

            if pd.isna(fill_value):
                fill_value = 0

            result[col] = result[col].fillna(fill_value)
            actions.append(
                f"Filled {missing_count} missing values in '{col}' using {missing_numeric}."
            )
        else:
            if missing_categorical == "Constant":
                fill_value = "Unknown"
            else:
                fill_value = safe_mode(result[col])

            result[col] = result[col].fillna(fill_value)
            actions.append(
                f"Filled {missing_count} missing values in '{col}' "
                f"using {missing_categorical}."
            )

    # Outlier detection/treatment on original numeric input features only.
    numeric_outlier_columns = [
        col for col in original_numeric_features
        if col in result.columns
    ]

    for col in numeric_outlier_columns:
        values = result[col]
        if values.dropna().nunique() < 4:
            outlier_report.append({
                "Column": col,
                "Outliers detected": 0,
                "Treatment": "Skipped: fewer than 4 unique values"
            })
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            outlier_report.append({
                "Column": col,
                "Outliers detected": 0,
                "Treatment": "Skipped: IQR is zero or unavailable"
            })
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (values < lower) | (values > upper)
        count = int(outlier_mask.sum())

        if outlier_method == "Keep original":
            treatment = "Detected; kept original values"
        elif count == 0:
            treatment = "No treatment required"
        elif outlier_method == "Remove outlier rows":
            result = result.loc[~outlier_mask].copy()
            treatment = "Removed outlier rows"
            actions.append(
                f"Removed {count} outlier rows based on IQR in '{col}'."
            )
        else:
            result[col] = result[col].clip(lower=lower, upper=upper)
            treatment = "Capped using IQR"
            actions.append(
                f"Capped {count} outlier values using IQR limits in '{col}'."
            )

        outlier_report.append({
            "Column": col,
            "Outliers detected": count,
            "Lower IQR limit": round(float(lower), 4),
            "Upper IQR limit": round(float(upper), 4),
            "Treatment": treatment
        })

    # Identify categorical feature columns after missing-value treatment.
    categorical_features = [
        col for col in result.columns
        if col != target_column and is_categorical(result[col])
    ]

    high_cardinality = detect_high_cardinality_columns(
        result,
        categorical_features,
        high_cardinality_threshold
    )

    if high_cardinality:
        actions.append(
            "High-cardinality categorical columns detected: "
            + ", ".join(high_cardinality)
        )

    # Choose which categorical columns to encode.
    if high_cardinality_method == "One-Hot encode all":
        columns_to_encode = categorical_features
        columns_to_leave = []
    elif high_cardinality_method == "Drop high-cardinality columns":
        columns_to_encode = [
            col for col in categorical_features if col not in high_cardinality
        ]
        columns_to_leave = []
        if high_cardinality:
            result = result.drop(columns=high_cardinality)
            actions.append(
                "Dropped high-cardinality columns: " + ", ".join(high_cardinality)
            )
    else:
        columns_to_encode = [
            col for col in categorical_features if col not in high_cardinality
        ]
        columns_to_leave = high_cardinality

    # Encode categorical feature columns only.
    if columns_to_encode:
        if encoding_method == "Label Encoding":
            for col in columns_to_encode:
                values = result[col].astype("string")
                categories = sorted(values.dropna().unique().tolist())
                mapping = {value: index for index, value in enumerate(categories)}
                result[col] = values.map(mapping).fillna(-1).astype(int)
                actions.append(f"Applied Label Encoding to '{col}'.")
        else:
            result = pd.get_dummies(
                result,
                columns=columns_to_encode,
                dtype=int
            )
            actions.append(
                "Applied One-Hot Encoding to: " + ", ".join(columns_to_encode)
            )

    if columns_to_leave:
        actions.append(
            "Kept high-cardinality columns unchanged: "
            + ", ".join(columns_to_leave)
        )

    # CRITICAL FIX:
    # Scale ONLY original numeric input columns.
    # Do not scale One-Hot Encoded columns, IDs, or the target.
    scale_columns = [
        col for col in original_numeric_features
        if col in result.columns
        and pd.api.types.is_numeric_dtype(result[col])
    ]

    if scale_features and scale_columns:
        scaler = (
            StandardScaler()
            if scale_features == "StandardScaler"
            else MinMaxScaler()
        )
        # StandardScaler and MinMaxScaler return floating-point values.
        # Convert integer columns to float before assignment to avoid:
        # "Invalid value ... for dtype int64".
        result[scale_columns] = result[scale_columns].astype("float64")
        result[scale_columns] = scaler.fit_transform(
            result[scale_columns]
        )
        actions.append(
            f"Applied {scale_features} only to original numeric features: "
            + ", ".join(scale_columns)
        )
    elif scale_features and not scale_columns:
        actions.append("Scaling was requested, but no eligible numeric features were found.")
    else:
        actions.append(
            f"Skipped feature scaling for {model_name} based on the selected settings."
        )

    # Keep target column last and unchanged.
    target_values = result[target_column].copy()
    result = result.drop(columns=[target_column])
    result[target_column] = target_values

    if not actions:
        actions.append("No cleaning changes were required.")

    summary = {
        "Original rows": original_rows,
        "Final rows": len(result),
        "Rows removed": original_rows - len(result),
        "Original columns": original_columns,
        "Final columns": result.shape[1],
        "Missing values before": int(source_df.isna().sum().sum()),
        "Missing values after": int(result.isna().sum().sum()),
        "Duplicates before": int(source_df.duplicated().sum()),
        "Duplicates after": int(result.duplicated().sum()),
        "Outlier columns checked": len(outlier_report),
        "Total outliers detected": sum(
            item.get("Outliers detected", 0) for item in outlier_report
        ),
    }

    return result, actions, summary, outlier_report


# =========================================================
# 5. EXPORT HELPERS
# =========================================================

def dataframe_to_excel(dataframe):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Cleaned Data")
    return output.getvalue()


def make_pdf_report(summary, actions, model_name, target_column, outlier_report):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        )

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()

        story = [
            Paragraph("CleanML - Dataset Cleaning Report", styles["Title"]),
            Spacer(1, 0.2 * inch),
            Paragraph(f"Selected algorithm: {escape(str(model_name))}", styles["Normal"]),
            Paragraph(f"Target column: {escape(str(target_column))}", styles["Normal"]),
            Spacer(1, 0.2 * inch),
            Paragraph("Dataset Summary", styles["Heading2"]),
        ]

        table_data = [["Metric", "Before", "After"]]
        table_data.extend([
            ["Rows", str(summary["Original rows"]), str(summary["Final rows"])],
            ["Columns", str(summary["Original columns"]), str(summary["Final columns"])],
            ["Missing values", str(summary["Missing values before"]), str(summary["Missing values after"])],
            ["Duplicate rows", str(summary["Duplicates before"]), str(summary["Duplicates after"])],
        ])

        table = Table(table_data, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7),
        ]))
        story.extend([table, Spacer(1, 0.2 * inch)])

        story.append(Paragraph("Outlier Detection", styles["Heading2"]))
        if outlier_report:
            outlier_table = [["Column", "Detected", "Treatment"]]
            for item in outlier_report:
                outlier_table.append([
                    str(item["Column"]),
                    str(item["Outliers detected"]),
                    str(item["Treatment"])
                ])
            table2 = Table(outlier_table, repeatRows=1, hAlign="LEFT")
            table2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(table2)
        else:
            story.append(Paragraph("No eligible numeric feature columns.", styles["Normal"]))

        story.extend([
            Spacer(1, 0.2 * inch),
            Paragraph("Cleaning Operations", styles["Heading2"])
        ])

        for action in actions:
            safe_action = escape(str(action))
            story.append(Paragraph(f"- {safe_action}", styles["Normal"]))
            story.append(Spacer(1, 0.05 * inch))

        doc.build(story)
        return buffer.getvalue()

    except ImportError:
        return None


# =========================================================
# 6. PAGE DESIGN
# =========================================================

st.markdown("""
<style>
/* ===== CleanML arcade-inspired interface ===== */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --cm-blue:#6d5dfc; --cm-cyan:#22d3ee; --cm-pink:#f472b6; --cm-ink:#172554; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp {
  background: radial-gradient(ellipse at 8% 4%, rgba(109,93,252,.13), transparent 34%),
              radial-gradient(ellipse at 92% 10%, rgba(34,211,238,.12), transparent 30%),
              linear-gradient(135deg, #f8faff 0%, #f2f5ff 48%, #fdf4ff 100%);
}
.block-container {
  padding-top: 4.5rem !important;
  padding-bottom: 3rem !important;
  max-width: 1450px;
}
[data-testid="stAppViewContainer"] .main .block-container {
  padding-top: 4.5rem !important;
}
p.main-title {
  display: block;
  position: relative;
  overflow: visible !important;
  padding-top: .15em;
  margin: 0 0 .35rem !important;
  font-family:'Space Grotesk',sans-serif;
  font-size:clamp(2.8rem,6vw,4.5rem);
  line-height:1.2;
  font-weight:700;
  letter-spacing:-2px;
  background:linear-gradient(90deg,#5546e8,#0891b2,#db4a9b,#5546e8);
  background-size:250% auto; -webkit-background-clip:text; background-clip:text;
  color:transparent; animation:gradientFlow 7s ease infinite;
}
.subtitle { font-size:1.08rem; color:#596782; margin:.35rem 0 1rem; }
@keyframes gradientFlow { 0%{background-position:0% center} 100%{background-position:250% center} }
@keyframes floatIn { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
@keyframes glowPulse { 0%,100%{box-shadow:0 8px 24px rgba(109,93,252,.16)} 50%{box-shadow:0 12px 34px rgba(34,211,238,.28)} }
@keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
.hero-panel {
  position:relative; overflow:hidden; padding:1.45rem 1.7rem; margin:.7rem 0 1.2rem;
  border-radius:22px; color:white;
  background:linear-gradient(115deg,#4f46e5 0%,#6d5dfc 45%,#0891b2 100%);
  box-shadow:0 16px 38px rgba(79,70,229,.20); animation:floatIn .7s ease both;
}
.hero-panel:after { content:''; position:absolute; width:240px;height:240px;right:-55px;top:-125px;
  border:1px solid rgba(255,255,255,.25);border-radius:50%;box-shadow:0 0 0 25px rgba(255,255,255,.06),0 0 0 55px rgba(255,255,255,.04); }
.hero-kicker {font-size:.76rem;letter-spacing:2px;text-transform:uppercase;opacity:.82;font-weight:700;}
.hero-copy {font-family:'Space Grotesk',sans-serif;font-size:1.45rem;font-weight:700;margin:.25rem 0;}
.hero-small {font-size:.9rem;opacity:.88;}
.step-chip {display:inline-block;padding:7px 12px;margin:3px;border-radius:999px;
  background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.24);font-size:.82rem;font-weight:600;}
div[data-testid="stMetric"] {
  background:rgba(255,255,255,.86); border:1px solid rgba(109,93,252,.16);
  padding:15px 16px; border-radius:17px; box-shadow:0 7px 22px rgba(31,41,95,.055);
  transition:transform .22s ease, box-shadow .22s ease; animation:floatIn .55s ease both;
}
div[data-testid="stMetric"]:hover { transform:translateY(-4px);box-shadow:0 13px 28px rgba(109,93,252,.14); }
div[data-testid="stMetric"] label {color:#66718b!important;font-weight:600!important;}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#3730a3;font-family:'Space Grotesk',sans-serif;}
[data-testid="stExpander"] {background:rgba(255,255,255,.72);border:1px solid rgba(109,93,252,.16);border-radius:16px;overflow:hidden;}
[data-testid="stFileUploader"] section {border:2px dashed #9b91f8;border-radius:18px;background:rgba(255,255,255,.68);transition:.2s ease;}
[data-testid="stFileUploader"] section:hover {border-color:#06b6d4;background:#f0fdff;}
.stButton>button, .stDownloadButton>button {
  border:0!important;border-radius:13px!important;padding:.65rem 1.1rem!important;
  font-weight:700!important;transition:transform .18s ease,filter .18s ease,box-shadow .18s ease!important;
  box-shadow:0 7px 18px rgba(79,70,229,.14)!important;
}
.stButton>button[kind="primary"] {background:linear-gradient(100deg,#5b4bea,#0e9fbd)!important;color:white!important;animation:glowPulse 3.2s ease-in-out infinite;}
.stButton>button:hover, .stDownloadButton>button:hover {transform:translateY(-3px);filter:brightness(1.04);box-shadow:0 12px 25px rgba(79,70,229,.2)!important;}
.stProgress>div>div>div>div {background:linear-gradient(90deg,#6d5dfc,#22d3ee,#f472b6);background-size:200% 100%;animation:shimmer 3s linear infinite;}
[data-testid="stDataFrame"] {border:1px solid rgba(109,93,252,.14);border-radius:14px;overflow:hidden;}
hr {border-color:rgba(109,93,252,.15)!important;}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#eef2ff,#f0fdfa);}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important;}}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🧹 CleanML</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Your data-cleaning playground — turn messy data into model-ready data.</p>',
    unsafe_allow_html=True
)
st.markdown("""
<div class="hero-panel">
  <div class="hero-kicker">LEVEL UP YOUR DATA</div>
  <div class="hero-copy">Ready, set, clean! ✨</div>
  <div class="hero-small">Upload your dataset, choose your ML algorithm, clean smartly and unlock your download.</div>
  <div style="margin-top:13px">
    <span class="step-chip">📂 Upload</span><span class="step-chip">🤖 Choose model</span>
    <span class="step-chip">🧹 Clean</span><span class="step-chip">🏆 Download</span>
  </div>
</div>
""", unsafe_allow_html=True)
st.caption("A little progress bar for your data journey: inspect → preprocess → compare → export")
st.divider()


# =========================================================
# 7. STEP 1 - UPLOAD DATASET
# =========================================================

st.header("1️⃣ Upload Your Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, XLSX and XLS"
)

if uploaded_file is None:
    st.info("Upload a dataset to begin.")
    st.markdown(
        "**CleanML workflow:** Upload → Choose algorithm → Choose target → "
        "Clean → Customize (optional) → View results → Download"
    )
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    dataset_key = hashlib.md5(file_bytes).hexdigest()

    if uploaded_file.name.lower().endswith(".csv"):
        try:
            uploaded_df = pd.read_csv(io.BytesIO(file_bytes))
        except UnicodeDecodeError:
            uploaded_df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
    elif uploaded_file.name.lower().endswith(".xls"):
        uploaded_df = pd.read_excel(io.BytesIO(file_bytes), engine="xlrd")
    else:
        uploaded_df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

    if uploaded_df.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()

    if uploaded_df.shape[1] < 2:
        st.error("Please upload at least two columns: features and a target.")
        st.stop()

    if uploaded_df.columns.duplicated().any():
        st.error("Duplicate column names found. Rename them and upload again.")
        st.stop()

    if st.session_state["dataset_key"] != dataset_key:
        reset_dataset_state(dataset_key, uploaded_df)

    original_df = st.session_state["original_data"].copy()

except Exception as error:
    st.error(f"Unable to read this dataset: {error}")
    st.stop()

st.success(f"'{uploaded_file.name}' uploaded successfully.")

with st.expander("👀 Preview uploaded dataset", expanded=True):
    st.dataframe(original_df.head(10), use_container_width=True)

overview = st.columns(4)
overview[0].metric("Rows", f"{len(original_df):,}")
overview[1].metric("Columns", f"{original_df.shape[1]:,}")
overview[2].metric("Missing cells", f"{int(original_df.isna().sum().sum()):,}")
overview[3].metric("Duplicate rows", f"{int(original_df.duplicated().sum()):,}")

numeric_count = len(original_df.select_dtypes(include=np.number).columns)
categorical_count = sum(is_categorical(original_df[col]) for col in original_df.columns)
profile = st.columns(2)
profile[0].metric("Numerical columns", numeric_count)
profile[1].metric("Categorical columns", categorical_count)

with st.expander("🔍 Data quality details"):
    quality = pd.DataFrame({
        "Column": original_df.columns.astype(str),
        "Data type": original_df.dtypes.astype(str).values,
        "Missing values": original_df.isna().sum().values,
        "Missing (%)": original_df.isna().mean().mul(100).round(2).values,
        "Unique values": original_df.nunique(dropna=True).values
    })
    st.dataframe(quality, use_container_width=True)

    missing_by_column = original_df.isna().sum()
    missing_by_column = missing_by_column[missing_by_column > 0]
    if not missing_by_column.empty:
        missing_chart = missing_by_column.rename_axis("Column").reset_index(
            name="Missing values"
        )
        fig = px.bar(
            missing_chart,
            x="Column",
            y="Missing values",
            title="Missing Values by Column"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found.")


# =========================================================
# 8. STEP 2 - CHOOSE ML ALGORITHM
# =========================================================

st.divider()
st.header("2️⃣ Choose Your Machine Learning Algorithm")

selected_model = st.selectbox(
    "Which algorithm are you preparing your data for?",
    MODEL_OPTIONS,
    key=f"model_{dataset_key}"
)

if selected_model in SCALE_MODELS:
    st.info(
        f"{selected_model} generally benefits from feature scaling. "
        "CleanML will scale eligible original numeric input columns only."
    )
else:
    st.info(
        f"{selected_model} generally does not require feature scaling. "
        "CleanML will leave numeric feature scales unchanged by default."
    )


# =========================================================
# 9. STEP 3 - CHOOSE TARGET COLUMN
# =========================================================

st.divider()
st.header("3️⃣ Choose Target / Output Column")

target_column = st.selectbox(
    "Select the column you want to predict:",
    options=original_df.columns.tolist(),
    key=f"target_{dataset_key}"
)

problem_type = get_problem_type(selected_model, original_df[target_column])

info1, info2, info3 = st.columns(3)
info1.metric("Selected algorithm", selected_model)
info2.metric("Target column", target_column)
info3.metric("Estimated problem type", problem_type)

st.caption(
    "The target column is protected from imputation, encoding, outlier treatment, "
    "and scaling. Rows with missing target values are removed because they have "
    "no known label for supervised learning."
)


# =========================================================
# 10. CLEANING SETTINGS
# =========================================================

auto_scale_method = (
    "StandardScaler" if selected_model in SCALE_MODELS else None
)

st.divider()
st.header("4️⃣ Clean Your Data Automatically")
st.write(
    "CleanML will remove duplicate records, fill missing feature values, "
    "inspect numeric feature outliers, encode categorical features, and "
    "scale original numeric features when appropriate."
)

if st.button("✨ Clean My Data", type="primary", use_container_width=True):
    try:
        with st.spinner("CleanML is analyzing and cleaning your dataset..."):
            cleaned, actions, summary, outlier_report = clean_dataset(
                source_df=original_df,
                target_column=target_column,
                model_name=selected_model,
                missing_numeric="Median",
                missing_categorical="Mode",
                outlier_method="Cap using IQR",
                encoding_method="One-Hot Encoding",
                scale_features=auto_scale_method,
                remove_duplicates=True,
                high_cardinality_method="Keep unchanged",
                high_cardinality_threshold=10
            )

        st.session_state["cleaned_data"] = cleaned
        st.session_state["cleaning_summary"] = summary
        st.session_state["cleaning_mode"] = "Automatic"
        st.session_state["last_clean_settings"] = {
            "model": selected_model,
            "target": target_column
        }
        st.session_state["outlier_report"] = outlier_report
        st.session_state["cleaning_history"] = []
        log_cleaning_actions("Automatic Cleaning", actions)

        st.success("Automatic cleaning completed successfully.")

    except Exception as error:
        st.error(f"Automatic cleaning failed: {error}")


# =========================================================
# 11. STEP 5 - OPTIONAL CUSTOMIZED CLEANING
# =========================================================

st.divider()
st.header("5️⃣ Customize Cleaning (Optional)")
st.write(
    "Choose your own cleaning methods. Customized cleaning always starts "
    "from the original uploaded dataset."
)

with st.expander("⚙️ Open customized cleaning options", expanded=False):
    st.subheader("Missing-value settings")

    custom_missing_numeric = st.selectbox(
        "Numerical missing values",
        ["Median", "Mean", "Zero"],
        key=f"custom_num_{dataset_key}"
    )

    custom_missing_categorical = st.selectbox(
        "Categorical missing values",
        ["Mode", "Constant (Unknown)"],
        key=f"custom_cat_{dataset_key}"
    )

    st.subheader("Outlier settings")
    custom_outliers = st.selectbox(
        "Outlier treatment",
        ["Cap using IQR", "Remove outlier rows", "Keep original"],
        key=f"custom_outlier_{dataset_key}"
    )

    st.subheader("Categorical encoding")
    custom_encoding = st.selectbox(
        "Encoding method",
        ["One-Hot Encoding", "Label Encoding"],
        key=f"custom_encoding_{dataset_key}"
    )

    high_cardinality_method = st.selectbox(
        "How should high-cardinality categorical columns be handled?",
        [
            "Keep unchanged",
            "One-Hot encode all",
            "Drop high-cardinality columns"
        ],
        index=0,
        key=f"custom_high_card_{dataset_key}"
    )

    high_cardinality_threshold = st.number_input(
        "High-cardinality threshold (unique values)",
        min_value=2,
        max_value=1000,
        value=10,
        step=1,
        key=f"custom_high_threshold_{dataset_key}"
    )

    st.subheader("Numerical scaling")
    custom_scaling = st.selectbox(
        "Scaling method",
        [
            "Automatic based on algorithm",
            "StandardScaler",
            "MinMaxScaler",
            "No scaling"
        ],
        key=f"custom_scaling_{dataset_key}"
    )

    custom_remove_duplicates = st.checkbox(
        "Remove duplicate rows",
        value=True,
        key=f"custom_duplicates_{dataset_key}"
    )

    st.caption(
        "Only original numeric feature columns are scaled. One-Hot Encoded columns, "
        "likely identifier columns, and the target column are not scaled."
    )

    if st.button("🛠️ Apply Customized Cleaning", use_container_width=True):
        try:
            if custom_scaling == "Automatic based on algorithm":
                scale_choice = auto_scale_method
            elif custom_scaling == "No scaling":
                scale_choice = None
            else:
                scale_choice = custom_scaling

            categorical_choice = (
                "Mode"
                if custom_missing_categorical == "Mode"
                else "Constant"
            )

            with st.spinner("Applying your customized cleaning settings..."):
                custom_cleaned, custom_actions, custom_summary, custom_outlier_report = clean_dataset(
                    source_df=original_df,
                    target_column=target_column,
                    model_name=selected_model,
                    missing_numeric=custom_missing_numeric,
                    missing_categorical=categorical_choice,
                    outlier_method=custom_outliers,
                    encoding_method=custom_encoding,
                    scale_features=scale_choice,
                    remove_duplicates=custom_remove_duplicates,
                    high_cardinality_method=high_cardinality_method,
                    high_cardinality_threshold=int(high_cardinality_threshold)
                )

            st.session_state["cleaned_data"] = custom_cleaned
            st.session_state["cleaning_summary"] = custom_summary
            st.session_state["cleaning_mode"] = "Customized"
            st.session_state["last_clean_settings"] = {
                "model": selected_model,
                "target": target_column
            }
            st.session_state["outlier_report"] = custom_outlier_report
            st.session_state["cleaning_history"] = []
            log_cleaning_actions("Customized Cleaning", custom_actions)

            st.success("Customized cleaning completed successfully.")

        except Exception as error:
            st.error(f"Customized cleaning failed: {error}")


# =========================================================
# 12. STEP 6 - RESULTS AND DOWNLOADS
# =========================================================

last_settings = st.session_state.get("last_clean_settings") or {}
results_match_selection = (
    last_settings.get("model") == selected_model
    and last_settings.get("target") == target_column
)

if (
    st.session_state["cleaned_data"] is not None
    and results_match_selection
):
    final_df = st.session_state["cleaned_data"].copy()
    summary = st.session_state["cleaning_summary"]
    history = st.session_state["cleaning_history"]
    outlier_report = st.session_state.get("outlier_report", [])

    st.divider()
    st.header("6️⃣ Cleaned Dataset Results")

    st.success(
        f"{st.session_state['cleaning_mode']} cleaning completed. "
        "Review the results and download your cleaned dataset."
    )

    metrics = st.columns(4)
    metrics[0].metric(
        "Rows remaining",
        f"{len(final_df):,}",
        delta=f"{len(final_df) - len(original_df):,}"
    )
    metrics[1].metric("Final columns", f"{len(final_df.columns):,}")
    metrics[2].metric(
        "Missing cells",
        f"{int(final_df.isna().sum().sum()):,}"
    )
    metrics[3].metric(
        "Duplicates",
        f"{int(final_df.duplicated().sum()):,}"
    )

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(final_df.head(20), use_container_width=True)

    st.subheader("Before-and-After Comparison")
    before, after = st.columns(2)

    with before:
        st.markdown("**Original Dataset**")
        st.write(f"Rows: {len(original_df):,}")
        st.write(f"Columns: {original_df.shape[1]:,}")
        st.write(f"Missing values: {int(original_df.isna().sum().sum()):,}")
        st.write(f"Duplicate rows: {int(original_df.duplicated().sum()):,}")

    with after:
        st.markdown("**Cleaned Dataset**")
        st.write(f"Rows: {len(final_df):,}")
        st.write(f"Columns: {final_df.shape[1]:,}")
        st.write(f"Missing values: {int(final_df.isna().sum().sum()):,}")
        st.write(f"Duplicate rows: {int(final_df.duplicated().sum()):,}")

    st.subheader("Outlier Detection Report")
    if outlier_report:
        st.dataframe(pd.DataFrame(outlier_report), use_container_width=True)
    else:
        st.info("No eligible numerical input columns were checked for outliers.")

    st.subheader("Cleaning Operations Performed")
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True)
    else:
        st.info("No cleaning actions recorded.")

    st.subheader("Download Your Results")
    csv_bytes = final_df.to_csv(index=False).encode("utf-8")
    excel_bytes = dataframe_to_excel(final_df)

    download1, download2 = st.columns(2)
    with download1:
        st.download_button(
            "⬇️ Download cleaned CSV",
            data=csv_bytes,
            file_name="CleanML_Cleaned_Dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

    with download2:
        st.download_button(
            "⬇️ Download cleaned Excel",
            data=excel_bytes,
            file_name="CleanML_Cleaned_Dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    pdf_bytes = make_pdf_report(
        summary=summary,
        actions=[item["Details"] for item in history],
        model_name=selected_model,
        target_column=target_column,
        outlier_report=outlier_report
    )

    if pdf_bytes is not None:
        st.download_button(
            "📄 Download cleaning report (PDF)",
            data=pdf_bytes,
            file_name="CleanML_Cleaning_Report.pdf",
            mime="application/pdf"
        )
    else:
        st.caption("Install reportlab to enable PDF cleaning reports.")

    if history:
        history_csv = pd.DataFrame(history).to_csv(index=False).encode("utf-8")
        st.download_button(
            "🕒 Download cleaning history (CSV)",
            data=history_csv,
            file_name="CleanML_Cleaning_History.csv",
            mime="text/csv"
        )

    st.info(
        "This application cleans and prepares datasets; it does not train or evaluate "
        "the selected ML model. For reliable model evaluation, fit preprocessing "
        "transformations on the training split only to avoid data leakage."
    )

elif st.session_state["cleaned_data"] is not None:
    st.warning(
        "You changed the selected algorithm or target. Run cleaning again to refresh results."
    )
else:
    st.info(
        "After selecting an algorithm and target column, click **Clean My Data** "
        "or open Customized Cleaning to choose your own settings."
    )


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()
st.caption(
    "CleanML | Smart data cleaning and ML preprocessing | "
    
)
