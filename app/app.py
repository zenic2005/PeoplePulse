from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PEOPLEPULSE - SINGLE FILE STREAMLIT APPLICATION
# ============================================================

st.set_page_config(
    page_title="PeoplePulse | Employee Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# app.py is inside the app folder
APP_DIR = Path(__file__).resolve().parent

# Move one level up to the main project folder
BASE_DIR = APP_DIR.parent

# Dataset paths
DATA_PATH = BASE_DIR / "data" / "cleaned_hr_data.csv"
FEATURE_DATA_PATH = BASE_DIR / "data" / "feature_engineered_hr_data.csv"

# Model paths
MODEL_PATH = BASE_DIR / "models" / "employee_attrition_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.csv"

HIGH_RISK = 0.60
MEDIUM_RISK = 0.30

HIGH_RISK_DEFAULT = HIGH_RISK
MEDIUM_RISK_DEFAULT = MEDIUM_RISK
# -------------------------- Theme ---------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: #090d16;
        color: #f8fafc;
        font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }
    #MainMenu, footer {visibility: hidden;}
    header {background: transparent !important;}
    .block-container {padding-top: 1.7rem; padding-bottom: 2rem;}
    .pp-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 13px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .kpi-title {
        color: #94a3b8; font-size: .76rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: .06em;
    }
    .kpi-value {
        color: #f8fafc; font-size: 1.85rem; font-weight: 750;
        margin-top: 7px; line-height: 1.15;
    }
    .kpi-note {color: #64748b; font-size: .76rem; margin-top: 7px;}
    .tag {
        display: inline-block; border-radius: 999px; padding: 4px 10px;
        font-size: .72rem; font-weight: 700; border: 1px solid;
    }
    .tag-high {color:#fb7185;background:#f43f5e18;border-color:#f43f5e55;}
    .tag-medium {color:#fbbf24;background:#f59e0b18;border-color:#f59e0b55;}
    .tag-low {color:#34d399;background:#10b98118;border-color:#10b98155;}
    .insight {
        background:#131c31; border-left:3px solid #38bdf8;
        padding:11px 14px; border-radius:0 8px 8px 0;
        margin:8px 0; color:#dbeafe; font-size:.88rem;
    }
    .muted {color:#94a3b8;}
    div[data-testid="stMetric"] {
        background:#0f172a; border:1px solid #1e293b;
        padding:14px; border-radius:12px;
    }
    div[data-testid="stMetric"] label {color:#94a3b8;}
    .stButton button[kind="primary"] {
        background:#0284c7; border:0; color:white; font-weight:700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="Inter, sans-serif"),
    margin=dict(l=18, r=18, t=38, b=18),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
)

# ---------------------- Data and model ----------------------

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_model(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    return joblib.load(path)


def normalize_attrition(series):
    """Return a consistent Yes/No representation for display."""
    return series.astype(str).str.strip().replace(
        {"1": "Yes", "0": "No", "True": "Yes", "False": "No"}
    )


def get_pipeline_feature_names(model):
    """Read the exact input schema stored by a fitted sklearn pipeline."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    # Some sklearn estimators expose feature names on the final step.
    for _, step in getattr(model, "steps", []):
        names = getattr(step, "feature_names_in_", None)
        if names is not None:
            return list(names)
    return None


def add_engineered_features(frame, reference_df=None):
    """
    Recreate the engineered columns used in the project notebook.
    Keep these definitions identical to notebook 04.
    """
    out = frame.copy()

    if "Age" in out.columns:
        out["AgeGroup"] = pd.cut(
            out["Age"], bins=[0, 25, 35, 45, 60],
            labels=["18-25", "26-35", "36-45", "46-60"],
            include_lowest=True,
        )

    if "MonthlyIncome" in out.columns:
        source = reference_df if reference_df is not None else out
        income_source = pd.to_numeric(
            source["MonthlyIncome"], errors="coerce"
        ).dropna()
        if len(income_source) >= 4:
            quantiles = income_source.quantile([0, .25, .50, .75, 1]).to_numpy()
            # Make quantile edges strictly increasing for duplicate-value safety.
            edges = np.unique(quantiles)
            if len(edges) >= 3:
                labels = ["Low", "Medium", "High", "Very High"][: len(edges) - 1]
                out["IncomeGroup"] = pd.cut(
                    out["MonthlyIncome"], bins=edges, labels=labels,
                    include_lowest=True, duplicates="drop",
                )
            else:
                out["IncomeGroup"] = "Medium"
        else:
            out["IncomeGroup"] = "Medium"

    if "YearsAtCompany" in out.columns:
        out["TenureGroup"] = pd.cut(
            out["YearsAtCompany"],
            bins=[-1, 2, 5, 10, 20, np.inf],
            labels=["0-2 Years", "3-5 Years", "6-10 Years", "11-20 Years", "20+ Years"],
        )

    if "YearsSinceLastPromotion" in out.columns:
        out["RecentlyPromoted"] = (
            pd.to_numeric(out["YearsSinceLastPromotion"], errors="coerce") <= 2
        ).astype(int)

    if "JobLevel" in out.columns:
        out["JobLevelGroup"] = pd.cut(
            out["JobLevel"], bins=[0, 1, 3, 5],
            labels=["Entry", "Mid", "Senior"], include_lowest=True,
        )

    satisfaction_columns = [
        "EnvironmentSatisfaction",
        "JobSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]
    available = [c for c in satisfaction_columns if c in out.columns]
    if available:
        out["OverallSatisfaction"] = out[available].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)

    if {"YearsAtCompany", "TotalWorkingYears"}.issubset(out.columns):
        total = pd.to_numeric(out["TotalWorkingYears"], errors="coerce").replace(0, np.nan)
        tenure = pd.to_numeric(out["YearsAtCompany"], errors="coerce")
        out["CompanyExperienceRatio"] = (tenure / total).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

    return out


def prepare_model_frame(raw_frame, model, reference_df):
    """Align raw/engineered records to the model's fitted feature schema."""
    frame = add_engineered_features(raw_frame, reference_df)
    feature_names = get_pipeline_feature_names(model)

    if feature_names is None:
        # Fallback for models that do not retain feature_names_in_.
        excluded = {
            "Attrition", "Attrition_Encoded", "EmployeeNumber", "EmployeeID",
            "EmployeeCount", "Over18", "StandardHours",
        }
        feature_names = [c for c in frame.columns if c not in excluded]

    # Add any expected columns from the reference data if available.
    for col in feature_names:
        if col not in frame.columns:
            if reference_df is not None and col in reference_df.columns:
                if pd.api.types.is_numeric_dtype(reference_df[col]):
                    frame[col] = reference_df[col].median()
                else:
                    mode = reference_df[col].mode(dropna=True)
                    frame[col] = mode.iloc[0] if not mode.empty else "Unknown"
            else:
                frame[col] = 0

    return frame[feature_names]


def positive_class_probability(model, X):
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The saved model does not support predict_proba(). "
            "Retrain/export a probability-capable classifier."
        )
    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        positive_index = classes.index(1)
    elif "Yes" in classes:
        positive_index = classes.index("Yes")
    else:
        positive_index = probabilities.shape[1] - 1
    return probabilities[:, positive_index]


def risk_label(probability, high_threshold, medium_threshold):
    if probability >= high_threshold:
        return "High Risk"
    if probability >= medium_threshold:
        return "Medium Risk"
    return "Low Risk"


def add_risk_scores(data, model, reference_df, high_threshold, medium_threshold):
    if model is None:
        return data.copy()
    X = prepare_model_frame(data, model, reference_df)
    probabilities = positive_class_probability(model, X)
    scored = data.copy()
    scored["AttritionProbability"] = probabilities
    scored["RiskLevel"] = [
        risk_label(p, high_threshold, medium_threshold) for p in probabilities
    ]
    return scored


def render_kpi(title, value, note, accent="#38bdf8"):
    st.markdown(
        f"""
        <div class="pp-card" style="border-top:3px solid {accent};">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level):
    css = {
        "High Risk": "tag-high",
        "Medium Risk": "tag-medium",
        "Low Risk": "tag-low",
    }.get(level, "tag-low")
    return f'<span class="tag {css}">{level}</span>'


def attrition_rate_by(frame, column):
    return (
        frame.groupby(column, dropna=False)["Attrition"]
        .apply(lambda s: (normalize_attrition(s) == "Yes").mean() * 100)
        .reset_index(name="Attrition Rate (%)")
    )


def render_filters(data, key_prefix):
    c1, c2, c3, c4, c5 = st.columns(5)
    dept_values = ["All"] + sorted(data["Department"].dropna().astype(str).unique().tolist())
    role_values = ["All"] + sorted(data["JobRole"].dropna().astype(str).unique().tolist())
    gender_values = ["All"] + sorted(data["Gender"].dropna().astype(str).unique().tolist())
    travel_values = ["All"] + sorted(data["BusinessTravel"].dropna().astype(str).unique().tolist())
    overtime_values = ["All"] + sorted(data["OverTime"].dropna().astype(str).unique().tolist())

    dept = c1.selectbox("Department", dept_values, key=f"{key_prefix}_dept")
    role = c2.selectbox("Job Role", role_values, key=f"{key_prefix}_role")
    gender = c3.selectbox("Gender", gender_values, key=f"{key_prefix}_gender")
    travel = c4.selectbox("Business Travel", travel_values, key=f"{key_prefix}_travel")
    overtime = c5.selectbox("Overtime", overtime_values, key=f"{key_prefix}_ot")

    result = data.copy()
    for column, selected in [
        ("Department", dept), ("JobRole", role), ("Gender", gender),
        ("BusinessTravel", travel), ("OverTime", overtime),
    ]:
        if selected != "All":
            result = result[result[column].astype(str) == selected]
    return result


# ----------------------- Load resources ---------------------

df = load_csv(str(DATA_PATH))
feature_df = load_csv(str(FEATURE_DATA_PATH))
model = load_model(str(MODEL_PATH))
metrics_df = load_csv(str(METRICS_PATH))

if df is None:
    st.error(
        "Could not find data/cleaned_hr_data.csv. Place your cleaned dataset "
        "inside the project's data folder."
    )
    st.stop()

if model is None:
    st.error(
        "Could not find models/employee_attrition_model.pkl. Export your fitted "
        "preprocessing + model pipeline from notebook 05 into the models folder."
    )
    st.stop()

if feature_df is None:
    feature_df = df.copy()

# Standardize target values only for dashboard display.
if "Attrition" not in df.columns:
    st.error("The cleaned dataset must contain an 'Attrition' column.")
    st.stop()

df["Attrition"] = normalize_attrition(df["Attrition"])
if "EmployeeNumber" in df.columns and "EmployeeID" not in df.columns:
    df["EmployeeID"] = df["EmployeeNumber"].apply(lambda x: f"EMP-{int(x):04d}")

if "EmployeeID" not in df.columns:
    df["EmployeeID"] = [f"EMP-{1001+i}" for i in range(len(df))]

# Keep the training reference table for imputing optional form fields.
reference_for_features = feature_df.copy()

if "high_risk_threshold" not in st.session_state:
    st.session_state.high_risk_threshold = HIGH_RISK_DEFAULT
if "medium_risk_threshold" not in st.session_state:
    st.session_state.medium_risk_threshold = MEDIUM_RISK_DEFAULT

try:
    scored_df = add_risk_scores(
        df, model, reference_for_features,
        st.session_state.high_risk_threshold,
        st.session_state.medium_risk_threshold,
    )
except Exception as exc:
    st.error(f"Unable to score the employee dataset with the saved model: {exc}")
    st.info(
        "Check that the model was trained with the same feature names and "
        "feature-engineering definitions used in this application."
    )
    st.stop()

# ---------------------- Sidebar navigation ------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 0 18px;border-bottom:1px solid #1e293b;margin-bottom:16px;">
            <div style="font-size:1.4rem;font-weight:800;color:#f8fafc;">PeoplePulse
                <span style="font-size:.65rem;color:#38bdf8;">AI</span>
            </div>
            <div style="font-size:.76rem;color:#94a3b8;">Employee Intelligence Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    section = st.radio(
        "Navigation",
        [
            "Overview",
            "Attrition Analytics",
            "Employee Risk",
            "Prediction",
            "Model Performance",
            "Employees",
            "Settings",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("IBM HR Analytics dataset")
    st.caption(f"{len(df):,} employee records")
    st.markdown(
        "<div style='padding-top:45px;color:#64748b;font-size:.75rem;'>"
        "PeoplePulse • Portfolio Project</div>",
        unsafe_allow_html=True,
    )

# ----------------------- Top header --------------------------

head1, head2 = st.columns([3, 1])
with head1:
    st.markdown("# PeoplePulse")
    st.caption("AI-powered employee attrition analytics and decision support")
with head2:
    st.markdown(
        "<div style='text-align:right;padding-top:12px;'>"
        "<span class='tag tag-low'>● Model Loaded</span></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='muted' style='font-size:.78rem;margin-bottom:12px;'>"
    "Predictions are exploratory decision-support estimates, not certainty "
    "and not a basis for employment decisions.</div>",
    unsafe_allow_html=True,
)

# ============================================================
# 1. OVERVIEW
# ============================================================

if section == "Overview":
    st.subheader("Executive Overview")
    st.caption("A snapshot of workforce composition and observed attrition.")

    view = render_filters(df, "overview")
    if view.empty:
        st.warning("No employees match these filters.")
        st.stop()

    attrition_count = int((view["Attrition"] == "Yes").sum())
    attrition_rate = attrition_count / len(view) * 100
    risk_view = scored_df[scored_df["EmployeeID"].isin(view["EmployeeID"])]
    high_risk_count = int((risk_view["RiskLevel"] == "High Risk").sum())
    average_income = pd.to_numeric(view["MonthlyIncome"], errors="coerce").mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Total Employees", f"{len(view):,}", "Employees in selected cohort")
    with k2:
        render_kpi("Observed Attrition Rate", f"{attrition_rate:.1f}%", f"{attrition_count:,} records marked Yes", "#f43f5e")
    with k3:
        render_kpi("Predicted High Risk", f"{high_risk_count:,}", "Model score above configured threshold", "#f59e0b")
    with k4:
        render_kpi("Average Monthly Income", f"₹{average_income:,.0f}", "Average in selected cohort", "#10b981")

    left, right = st.columns(2)
    with left:
        st.markdown("<div class='pp-card'><b>Observed Attrition Distribution</b>", unsafe_allow_html=True)
        counts = view["Attrition"].value_counts().reindex(["No", "Yes"], fill_value=0).reset_index()
        counts.columns = ["Attrition", "Employees"]
        fig = px.pie(
            counts, names="Attrition", values="Employees", hole=.62,
            color="Attrition", color_discrete_map={"No": "#10b981", "Yes": "#f43f5e"},
        )
        fig.update_layout(**PLOT_LAYOUT, height=330)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='pp-card'><b>Attrition Rate by Department</b>", unsafe_allow_html=True)
        dep = attrition_rate_by(view, "Department")
        fig = px.bar(dep, x="Department", y="Attrition Rate (%)", text_auto=".1f",
                     color_discrete_sequence=["#38bdf8"])
        fig.update_layout(**PLOT_LAYOUT, height=330)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("<div class='pp-card'><b>Attrition by Job Role</b>", unsafe_allow_html=True)
        role = attrition_rate_by(view, "JobRole").sort_values("Attrition Rate (%)")
        fig = px.bar(role, x="Attrition Rate (%)", y="JobRole", orientation="h",
                     color_discrete_sequence=["#a78bfa"])
        fig.update_layout(**PLOT_LAYOUT, height=430)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='pp-card'><b>Overtime vs Attrition</b>", unsafe_allow_html=True)
        ot = attrition_rate_by(view, "OverTime")
        fig = px.bar(ot, x="OverTime", y="Attrition Rate (%)", text_auto=".1f",
                     color_discrete_sequence=["#fb923c"])
        fig.update_layout(**PLOT_LAYOUT, height=430)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 2. ATTRITION ANALYTICS
# ============================================================

elif section == "Attrition Analytics":
    st.subheader("Attrition Analytics")
    st.caption("Explore historical attrition patterns across employee attributes.")
    view = render_filters(df, "analytics")
    if view.empty:
        st.warning("No employees match these filters.")
        st.stop()

    st.markdown(
        "<div class='insight'>Explore these charts as historical associations. "
        "They do not prove that any individual attribute causes attrition.</div>",
        unsafe_allow_html=True,
    )

    view = view.copy()
    view["Age Group"] = pd.cut(
        view["Age"], bins=[17, 25, 35, 45, 60],
        labels=["18–25", "26–35", "36–45", "46–60"],
    )
    view["Tenure Group"] = pd.cut(
        view["YearsAtCompany"], bins=[-1, 2, 5, 10, 20, np.inf],
        labels=["0–2", "3–5", "6–10", "11–20", "20+"],
    )

    chart_specs = [
        ("Age Group", "Attrition Rate by Age Group"),
        ("Tenure Group", "Attrition Rate by Years at Company"),
        ("JobSatisfaction", "Attrition Rate by Job Satisfaction"),
        ("MaritalStatus", "Attrition Rate by Marital Status"),
        ("BusinessTravel", "Attrition Rate by Business Travel"),
        ("JobInvolvement", "Attrition Rate by Job Involvement"),
    ]
    for start in range(0, len(chart_specs), 2):
        cols = st.columns(2)
        for col, (feature, title) in zip(cols, chart_specs[start:start+2]):
            with col:
                st.markdown(f"<div class='pp-card'><b>{title}</b>", unsafe_allow_html=True)
                result = attrition_rate_by(view, feature)
                result[feature] = result[feature].astype(str)
                fig = px.bar(
                    result, x=feature, y="Attrition Rate (%)",
                    text_auto=".1f", color_discrete_sequence=["#38bdf8"],
                )
                fig.update_layout(**PLOT_LAYOUT, height=330)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='pp-card'><b>Monthly Income by Observed Attrition</b>", unsafe_allow_html=True)
    fig = px.box(
        view, x="Attrition", y="MonthlyIncome", color="Attrition",
        color_discrete_map={"No": "#10b981", "Yes": "#f43f5e"},
    )
    fig.update_layout(**PLOT_LAYOUT, height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 3. EMPLOYEE RISK
# ============================================================

elif section == "Employee Risk":
    st.subheader("Employee Risk Dashboard")
    st.caption("Risk categories are derived from the saved model's predicted probabilities.")

    low = int((scored_df["RiskLevel"] == "Low Risk").sum())
    medium = int((scored_df["RiskLevel"] == "Medium Risk").sum())
    high = int((scored_df["RiskLevel"] == "High Risk").sum())

    a, b, c = st.columns(3)
    with a:
        render_kpi("Low Risk", f"{low:,}", f"Below {st.session_state.medium_risk_threshold:.0%}", "#10b981")
    with b:
        render_kpi("Medium Risk", f"{medium:,}", "Monitor and review", "#f59e0b")
    with c:
        render_kpi("High Risk", f"{high:,}", f"At or above {st.session_state.high_risk_threshold:.0%}", "#f43f5e")

    left, right = st.columns([1, 2])
    with left:
        fig = go.Figure(data=[go.Pie(
            labels=["Low Risk", "Medium Risk", "High Risk"],
            values=[low, medium, high], hole=.65,
            marker=dict(colors=["#10b981", "#f59e0b", "#f43f5e"]),
        )])
        fig.update_layout(**PLOT_LAYOUT, height=340, title="Predicted Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Highest predicted probabilities")
        risk_view = scored_df.sort_values("AttritionProbability", ascending=False).copy()
        c1, c2 = st.columns(2)
        query = c1.text_input("Search employee ID or job role", key="risk_query")
        tier = c2.selectbox("Risk category", ["All", "High Risk", "Medium Risk", "Low Risk"], key="risk_tier")
        if query:
            mask = (
                risk_view["EmployeeID"].astype(str).str.contains(query, case=False, na=False)
                | risk_view["JobRole"].astype(str).str.contains(query, case=False, na=False)
            )
            risk_view = risk_view[mask]
        if tier != "All":
            risk_view = risk_view[risk_view["RiskLevel"] == tier]
        display = risk_view[
            ["EmployeeID", "Department", "JobRole", "OverTime", "AttritionProbability", "RiskLevel"]
        ].copy()
        display["AttritionProbability"] = display["AttritionProbability"].map(lambda x: f"{x:.1%}")
        st.dataframe(display.head(100), use_container_width=True, hide_index=True)

# ============================================================
# 4. PREDICTION
# ============================================================

elif section == "Prediction":
    st.subheader("Individual Employee Prediction")
    st.caption("Enter employee information. The application passes the record through your saved preprocessing + ML pipeline.")

    # Values are taken from the actual dataset where possible.
    def numeric_default(column, fallback):
        if column in df.columns:
            value = pd.to_numeric(df[column], errors="coerce").median()
            return fallback if pd.isna(value) else int(round(value))
        return fallback

    def categorical_options(column, fallback):
        if column in df.columns:
            values = sorted(df[column].dropna().astype(str).unique().tolist())
            if values:
                return values
        return fallback

    with st.form("employee_prediction_form"):
        st.markdown("#### Personal information")
        p1, p2, p3, p4 = st.columns(4)
        age = p1.number_input("Age", 18, 65, numeric_default("Age", 30))
        gender_options = categorical_options("Gender", ["Male", "Female"])
        gender = p2.selectbox("Gender", gender_options)
        marital_options = categorical_options("MaritalStatus", ["Single", "Married", "Divorced"])
        marital = p3.selectbox("Marital Status", marital_options)
        distance = p4.number_input("Distance From Home", 1, 50, numeric_default("DistanceFromHome", 10))

        st.markdown("#### Job information")
        j1, j2, j3 = st.columns(3)
        dept_options = categorical_options("Department", ["Sales", "Research & Development", "Human Resources"])
        department = j1.selectbox("Department", dept_options)
        role_options = categorical_options("JobRole", ["Sales Executive"])
        job_role = j2.selectbox("Job Role", role_options)
        job_level = j3.slider("Job Level", 1, 5, min(5, max(1, numeric_default("JobLevel", 2))))

        j4, j5, j6 = st.columns(3)
        travel_options = categorical_options("BusinessTravel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        business_travel = j4.selectbox("Business Travel", travel_options)
        overtime_options = categorical_options("OverTime", ["Yes", "No"])
        overtime = j5.selectbox("Overtime", overtime_options)
        job_involvement = j6.slider("Job Involvement", 1, 4, min(4, max(1, numeric_default("JobInvolvement", 3))))

        st.markdown("#### Compensation")
        c1, c2, c3 = st.columns(3)
        monthly_income = c1.number_input("Monthly Income (₹)", 1000, 100000, max(1000, numeric_default("MonthlyIncome", 5000)), step=250)
        salary_hike = c2.slider("Percent Salary Hike", 10, 30, min(30, max(10, numeric_default("PercentSalaryHike", 15))))
        stock_option = c3.selectbox("Stock Option Level", [0, 1, 2, 3], index=min(3, max(0, numeric_default("StockOptionLevel", 1))))

        st.markdown("#### Experience")
        e1, e2, e3 = st.columns(3)
        total_years = e1.number_input("Total Working Years", 0, 50, numeric_default("TotalWorkingYears", 6))
        years_company = e2.number_input("Years at Company", 0, 50, numeric_default("YearsAtCompany", 3))
        years_role = e3.number_input("Years in Current Role", 0, 40, numeric_default("YearsInCurrentRole", 2))
        e4, e5, e6 = st.columns(3)
        years_promotion = e4.number_input("Years Since Last Promotion", 0, 40, numeric_default("YearsSinceLastPromotion", 1))
        years_manager = e5.number_input("Years With Current Manager", 0, 40, numeric_default("YearsWithCurrManager", 2))
        companies_worked = e6.number_input("Number of Companies Worked", 0, 20, numeric_default("NumCompaniesWorked", 1))

        st.markdown("#### Satisfaction and education")
        s1, s2, s3, s4 = st.columns(4)
        environment_sat = s1.slider("Environment Satisfaction", 1, 4, min(4, max(1, numeric_default("EnvironmentSatisfaction", 3))))
        job_sat = s2.slider("Job Satisfaction", 1, 4, min(4, max(1, numeric_default("JobSatisfaction", 3))))
        relationship_sat = s3.slider("Relationship Satisfaction", 1, 4, min(4, max(1, numeric_default("RelationshipSatisfaction", 3))))
        work_life = s4.slider("Work-Life Balance", 1, 4, min(4, max(1, numeric_default("WorkLifeBalance", 3))))

        s5, s6, s7 = st.columns(3)
        education = s5.selectbox("Education", [1, 2, 3, 4, 5], index=min(4, max(0, numeric_default("Education", 3)-1)))
        education_fields = categorical_options(
            "EducationField",
            ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"],
        )
        education_field = s6.selectbox("Education Field", education_fields)
        performance = s7.selectbox("Performance Rating", [1, 2, 3, 4], index=min(3, max(0, numeric_default("PerformanceRating", 3)-1)))

        submitted = st.form_submit_button(
            "Predict Attrition Risk", type="primary", use_container_width=True
        )

    if submitted:
        # Start with representative values for any model features not exposed
        # in this compact form, then replace all fields entered by the user.
        base = {}
        for col in df.columns:
            if col in {"Attrition", "EmployeeID", "EmployeeNumber", "AttritionProbability", "RiskLevel"}:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                base[col] = pd.to_numeric(df[col], errors="coerce").median()
            else:
                mode = df[col].mode(dropna=True)
                base[col] = mode.iloc[0] if not mode.empty else "Unknown"

        user_values = {
            "Age": age, "Gender": gender, "MaritalStatus": marital,
            "DistanceFromHome": distance, "Department": department,
            "JobRole": job_role, "JobLevel": job_level,
            "BusinessTravel": business_travel, "OverTime": overtime,
            "JobInvolvement": job_involvement, "MonthlyIncome": monthly_income,
            "PercentSalaryHike": salary_hike, "StockOptionLevel": stock_option,
            "TotalWorkingYears": total_years, "YearsAtCompany": years_company,
            "YearsInCurrentRole": years_role,
            "YearsSinceLastPromotion": years_promotion,
            "YearsWithCurrManager": years_manager,
            "NumCompaniesWorked": companies_worked,
            "EnvironmentSatisfaction": environment_sat,
            "JobSatisfaction": job_sat,
            "RelationshipSatisfaction": relationship_sat,
            "WorkLifeBalance": work_life,
            "Education": education, "EducationField": education_field,
            "PerformanceRating": performance,
        }
        base.update(user_values)
        input_raw = pd.DataFrame([base])

        try:
            X_input = prepare_model_frame(input_raw, model, reference_for_features)
            probability = float(positive_class_probability(model, X_input)[0])
            prediction = int(probability >= HIGH_RISK_DEFAULT)
            risk = risk_label(
                probability,
                st.session_state.high_risk_threshold,
                st.session_state.medium_risk_threshold,
            )

            st.markdown("---")
            result_col, detail_col = st.columns([1, 1.4])
            with result_col:
                st.markdown("### Prediction result")
                st.markdown(risk_badge(risk), unsafe_allow_html=True)
                st.metric("Estimated attrition probability", f"{probability:.1%}")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%", "font": {"color": "#f8fafc", "size": 38}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                        "bar": {"color": "#f43f5e" if risk == "High Risk" else "#f59e0b" if risk == "Medium Risk" else "#10b981"},
                        "bgcolor": "#1e293b",
                        "steps": [
                            {"range": [0, st.session_state.medium_risk_threshold * 100], "color": "#10b98122"},
                            {"range": [st.session_state.medium_risk_threshold * 100, st.session_state.high_risk_threshold * 100], "color": "#f59e0b22"},
                            {"range": [st.session_state.high_risk_threshold * 100, 100], "color": "#f43f5e22"},
                        ],
                    },
                ))
                fig.update_layout(**PLOT_LAYOUT, height=260)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Risk bands are configurable thresholds, not calibrated clinical or HR standards.")

            with detail_col:
                st.markdown("### Submitted employee profile")
                summary = pd.DataFrame(
                    list(user_values.items()), columns=["Feature", "Entered value"]
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)
                st.info(
                    "This estimate is generated by the saved ML pipeline. "
                    "It should be used only for aggregate research and supportive "
                    "retention discussions, never as the sole basis for an employment decision."
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.warning(
                "Most commonly, this means the feature names or feature-engineering "
                "logic in app.py do not exactly match notebook 04 and notebook 05."
            )

# ============================================================
# 5. MODEL PERFORMANCE
# ============================================================

elif section == "Model Performance":
    st.subheader("Model Performance")
    st.caption("Evaluation results exported from notebook 05.")

    if metrics_df is None:
        st.warning(
            "models/model_metrics.csv was not found. Export the comparison "
            "DataFrame from notebook 05 to this path."
        )
    else:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        numeric_metrics = [
            c for c in ["Accuracy", "Precision", "Recall", "F1", "F1 Score", "ROC-AUC", "PR-AUC"]
            if c in metrics_df.columns
        ]
        model_column = next(
            (c for c in ["Model", "model", "Algorithm", "Classifier"] if c in metrics_df.columns),
            None,
        )
        if numeric_metrics and model_column:
            chosen_metric = st.selectbox("Compare models by", numeric_metrics)
            plot_data = metrics_df[[model_column, chosen_metric]].copy()
            fig = px.bar(
                plot_data, x=model_column, y=chosen_metric,
                text_auto=".3f", color_discrete_sequence=["#38bdf8"],
            )
            fig.update_layout(**PLOT_LAYOUT, height=380)
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Only show ROC/PR curves and feature importance here if you also export "
            "the actual evaluation curve data or feature-importance results from notebook 05."
        )

# ============================================================
# 6. EMPLOYEE DIRECTORY
# ============================================================

elif section == "Employees":
    st.subheader("Employee Directory")
    st.caption("Search and inspect employee records from the cleaned dataset.")

    f1, f2, f3, f4 = st.columns([2, 1.2, 1.2, 1])
    query = f1.text_input("Search employee ID or role", key="directory_search")
    department_filter = f2.selectbox(
        "Department", ["All"] + sorted(df["Department"].dropna().astype(str).unique().tolist())
    )
    role_filter = f3.selectbox(
        "Job Role", ["All"] + sorted(df["JobRole"].dropna().astype(str).unique().tolist())
    )
    attrition_filter = f4.selectbox("Attrition", ["All", "Yes", "No"])

    view = scored_df.copy()
    if query:
        mask = (
            view["EmployeeID"].astype(str).str.contains(query, case=False, na=False)
            | view["JobRole"].astype(str).str.contains(query, case=False, na=False)
        )
        view = view[mask]
    if department_filter != "All":
        view = view[view["Department"] == department_filter]
    if role_filter != "All":
        view = view[view["JobRole"] == role_filter]
    if attrition_filter != "All":
        view = view[view["Attrition"] == attrition_filter]

    display_columns = [
        c for c in [
            "EmployeeID", "Age", "Department", "JobRole", "MonthlyIncome",
            "YearsAtCompany", "OverTime", "JobSatisfaction", "Attrition",
            "AttritionProbability", "RiskLevel",
        ] if c in view.columns
    ]
    table = view[display_columns].copy()
    if "AttritionProbability" in table:
        table["AttritionProbability"] = table["AttritionProbability"].map(lambda x: f"{x:.1%}")
    if "MonthlyIncome" in table:
        table["MonthlyIncome"] = table["MonthlyIncome"].map(lambda x: f"₹{x:,.0f}")
    st.dataframe(table, use_container_width=True, hide_index=True, height=420)

    if not view.empty:
        selected = st.selectbox("Inspect employee", view["EmployeeID"].astype(str).tolist())
        employee = view[view["EmployeeID"].astype(str) == selected].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Age", int(employee["Age"]))
        c2.metric("Department", str(employee["Department"]))
        c3.metric("Attrition", str(employee["Attrition"]))
        c4.metric("Predicted risk", str(employee["RiskLevel"]))
        st.caption(
            f"Predicted probability: {employee['AttritionProbability']:.1%}. "
            "The model score is an estimate, not a statement about an employee's intentions."
        )

# ============================================================
# 7. SETTINGS
# ============================================================

elif section == "Settings":
    st.subheader("Application Settings")
    st.caption("Configure display thresholds and inspect the loaded project assets.")

    st.markdown("#### Risk classification thresholds")
    medium = st.slider(
        "Medium-risk threshold",
        min_value=0.05, max_value=0.70,
        value=float(st.session_state.medium_risk_threshold), step=0.05,
    )
    high = st.slider(
        "High-risk threshold",
        min_value=max(0.30, medium + 0.05), max_value=0.95,
        value=max(float(st.session_state.high_risk_threshold), medium + 0.05),
        step=0.05,
    )
    if st.button("Apply thresholds", type="primary"):
        st.session_state.medium_risk_threshold = medium
        st.session_state.high_risk_threshold = high
        st.rerun()

    st.markdown("#### Project assets")
    asset_status = pd.DataFrame(
        [
            ["Cleaned dataset", DATA_PATH, DATA_PATH.exists()],
            ["Feature-engineered dataset", FEATURE_DATA_PATH, FEATURE_DATA_PATH.exists()],
            ["Trained model pipeline", MODEL_PATH, MODEL_PATH.exists()],
            ["Model metrics", METRICS_PATH, METRICS_PATH.exists()],
        ],
        columns=["Asset", "Path", "Available"],
    )
    asset_status["Path"] = asset_status["Path"].astype(str)
    st.dataframe(asset_status, use_container_width=True, hide_index=True)
    st.caption(f"Loaded model type: {type(model).__name__}")
    st.caption(f"Dataset records: {len(df):,}")

    st.markdown(
        "<div class='insight'>Important: risk thresholds change only the Low / Medium / "
        "High labels. They do not retrain the model or change its predicted probability.</div>",
        unsafe_allow_html=True,
    )

# --------------------------- Footer -------------------------

st.markdown("---")
st.caption(
    "PeoplePulse | Educational portfolio project. Historical patterns and model "
    "scores are not causal evidence and must not be used as the sole basis for "
    "employment decisions."
)
