import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

df = pd.read_csv("clean_property_data.csv")
# -----------------------------
# LOAD MODEL + ENCODERS
# -----------------------------
rf_model = joblib.load("rf_model.pkl")

property_encoder = joblib.load("property_encoder.pkl")
tenure_encoder = joblib.load("tenure_encoder.pkl")
district_encoder = joblib.load("district_encoder.pkl")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="UK Property Analytics Dashboard",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

/* MAIN BACKGROUND */
.main {
    background-color: #0B1020;
}

/* TEXT */
html, body, [class*="css"] {
    color: #F5F1E8;
    font-family: 'Segoe UI', sans-serif;
}

/* TITLES */
h1, h2, h3 {
    color: #F5F1E8;
    font-weight: 600;
}

/* KPI CARDS */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #161B2E, #1D2440);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* BUTTONS */
.stButton > button {
    background-color: #D4A017;
    color: black;
    border: none;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: 600;
    transition: 0.3s ease;
}

.stButton > button:hover {
    background-color: #E6B325;
    color: black;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #B8BCC8;
    font-size: 17px;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    color: #D4A017;
    border-bottom: 3px solid #D4A017;
}

/* CHART CONTAINERS */
.element-container {
    border-radius: 14px;
}

/* CAPTIONS */
.stCaption {
    color: #B8BCC8;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("clean_property_data.csv")

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filter Data")
st.sidebar.markdown(
    "Use the filters below to explore property market trends and predictions."
)

district = st.sidebar.selectbox(
    "Select District",
    ["All"] + sorted(df['district'].dropna().unique().tolist())
)

property_type = st.sidebar.selectbox(
    "Select Property Type",
    ["All"] + sorted(df['property_type'].dropna().unique().tolist())
)

year = st.sidebar.selectbox(
    "Select Year",
    ["All"] + sorted(df['year'].dropna().unique().tolist())
)

# -----------------------------
# FILTER DATA
# -----------------------------
filtered_df = df.copy()

if district != "All":
    filtered_df = filtered_df[filtered_df['district'] == district]

if property_type != "All":
    filtered_df = filtered_df[filtered_df['property_type'] == property_type]

if year != "All":
    filtered_df = filtered_df[filtered_df['year'] == year]

# -----------------------------
# DASHBOARD TITLE
# -----------------------------
st.markdown("""
<h1 style='text-align: center; color: white;'>
UK Property Price Analytics Dashboard
</h1>

<p style='text-align: center; font-size:18px; color: lightgray;'>
Interactive predictive analytics system for UK residential property investment decision-making
</p>
""", unsafe_allow_html=True)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "Market Analysis",
    "Prediction System",
    "Model Evaluation"
])

# =========================================================
# TAB 1 — MARKET ANALYSIS
# =========================================================
with tab1:

    # -----------------------------
    # KPI METRICS
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Property Price",
        f"£{filtered_df['price'].mean():,.0f}"
    )

    col2.metric(
        "Total Transactions",
        f"{len(filtered_df):,}"
    )

    col3.metric(
        "Highest Property Price",
        f"£{filtered_df['price'].max():,.0f}"
    )

    st.markdown("---")

    # -----------------------------
    # PRICE TREND
    # -----------------------------
    st.subheader("Property Price Trend Over Time")

    yearly_price = (
        filtered_df.groupby('year')['price']
        .mean()
        .reset_index()
    )

    fig1 = px.line(
        yearly_price,
        x='year',
        y='price',
        markers=True,
        template='plotly_dark'
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # TWO COLUMN LAYOUT
    # -----------------------------
    col4, col5 = st.columns(2)

    # PROPERTY TYPE ANALYSIS
    with col4:

        st.subheader("Average Price by Property Type")

        ptype = (
            filtered_df[filtered_df['property_type'] != 'O']
            .groupby('property_type')['price']
            .mean()
            .reset_index()
        )

        ptype['property_type'] = ptype['property_type'].replace({
            'F': 'Flat',
            'T': 'Terraced',
            'S': 'Semi-Detached',
            'D': 'Detached'
        })

        fig2 = px.bar(
            ptype,
            x='property_type',
            y='price',
            color='property_type',
            template='plotly_dark'
        )

        st.plotly_chart(fig2, use_container_width=True)

    # TENURE ANALYSIS
    with col5:

        st.subheader("Average Price by Tenure Type")

        tenure = (
            filtered_df.groupby('tenure')['price']
            .mean()
            .reset_index()
        )

        tenure['tenure'] = tenure['tenure'].replace({
            'F': 'Freehold',
            'L': 'Leasehold'
        })

        fig3 = px.pie(
            tenure,
            names='tenure',
            values='price',
            template='plotly_dark'
        )

        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # TOP DISTRICTS
    # -----------------------------
    st.subheader("Top 10 Districts by Average Property Price")

    district_price = (
        filtered_df.groupby('district')['price']
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig4 = px.bar(
        district_price,
        x='price',
        y='district',
        orientation='h',
        color='price',
        template='plotly_dark',
        color_continuous_scale='blues'
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # MONTHLY TREND
    # -----------------------------
    st.subheader("Average Property Price by Month")

    monthly_price = (
        filtered_df.groupby('month')['price']
        .mean()
        .reset_index()
    )

    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
        9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }

    monthly_price['month'] = monthly_price['month'].map(month_names)

    fig5 = px.line(
        monthly_price,
        x='month',
        y='price',
        markers=True,
        template='plotly_dark'
    )

    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # RISK ANALYSIS
    # -----------------------------
    st.subheader("Property Price Variability (Risk Analysis)")

    risk_df = (
        filtered_df[filtered_df['property_type'] != 'O']
        .groupby('property_type')['price']
        .agg('std')
        .reset_index(name='price_std')
    )

    risk_df['property_type'] = risk_df['property_type'].replace({
        'F': 'Flat',
        'T': 'Terraced',
        'S': 'Semi-Detached',
        'D': 'Detached'
    })

    fig7 = px.bar(
        risk_df,
        x='property_type',
        y='price_std',
        color='price_std',
        template='plotly_dark',
        color_continuous_scale='reds'
    )

    st.plotly_chart(fig7, use_container_width=True)

# =========================================================
# TAB 2 — PREDICTION SYSTEM
# =========================================================
with tab2:

    st.subheader("Predict Property Price")

    st.markdown(
        "Select property features to estimate the predicted property price."
    )

    col6, col7 = st.columns(2)

    with col6:

        property_options = {
            'Flat': 'F',
            'Terraced': 'T',
            'Semi-Detached': 'S',
            'Detached': 'D'
        }

        selected_property = st.selectbox(
            "Property Type",
            list(property_options.keys())
        )

        input_property = property_options[selected_property]

        tenure_options = {
            'Freehold': 'F',
            'Leasehold': 'L'
        }

        selected_tenure = st.selectbox(
            "Tenure Type",
            list(tenure_options.keys())
        )

        input_tenure = tenure_options[selected_tenure]

        input_year = st.selectbox(
            "Year",
            sorted(df['year'].dropna().astype(int).unique())
        )

    with col7:

        input_district = st.selectbox(
            "District",
            sorted(df['district'].dropna().unique())
        )

        input_month = st.selectbox(
            "Month",
            list(range(1, 13))
        )

    if st.button("Predict Price"):

        property_encoded = property_encoder.transform([input_property])[0]
        tenure_encoded = tenure_encoder.transform([input_tenure])[0]
        district_encoded = district_encoder.transform([input_district])[0]

        input_data = pd.DataFrame({
            'property_type_encoded': [property_encoded],
            'tenure_encoded': [tenure_encoded],
            'district_encoded': [district_encoded],
            'year': [input_year],
            'month': [input_month]
        })

        prediction = rf_model.predict(input_data)[0]

        st.success(
            f"Estimated Property Price: £{prediction:,.0f}"
        )

# =========================================================
# TAB 3 — MODEL EVALUATION
# =========================================================
with tab3:

    st.subheader("Model Performance Comparison")

    model_results = pd.DataFrame({
        'Model': [
            'Linear Regression',
            'Random Forest',
            'Gradient Boosting'
        ],
        'R² Score': [
            0.404,
            0.196,
            0.427
        ]
    })

    fig6 = px.bar(
        model_results,
        x='Model',
        y='R² Score',
        color='Model',
        template='plotly_dark',
color_discrete_sequence=['#D4A017', '#4F81BD'],
        text='R² Score'
    )

    fig6.update_traces(
        texttemplate='%{text:.3f}',
        textposition='outside'
    )

    st.plotly_chart(fig6, use_container_width=True)

# -----------------------------
# RISK INTERPRETATION
# -----------------------------
st.info(
    "Higher price variability indicates greater investment risk and uncertainty. "
    "Property types with lower variability demonstrate more stable pricing behaviour."
)