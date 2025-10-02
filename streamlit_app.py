import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("airline_accidents.csv", low_memory=False)

    # Normalize column names
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    # Standardize date column if available
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        df["year"] = df["event_date"].dt.year
    elif "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Convert injury columns to numeric
    injury_cols = ["total_fatal_injuries","total_serious_injuries","total_minor_injuries","total_uninjured"]
    for col in injury_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

df = load_data()

# -------------------------------
# Sidebar filters
# -------------------------------
st.sidebar.header("Filters")

make_options = df["make"].dropna().unique() if "make" in df.columns else []
selected_make = st.sidebar.multiselect("Select Aircraft Make:", make_options)

year_min = int(df["year"].min()) if "year" in df.columns else None
year_max = int(df["year"].max()) if "year" in df.columns else None
selected_years = st.sidebar.slider("Select Year Range:",
                                   min_value=year_min,
                                   max_value=year_max,
                                   value=(year_min, year_max)) if year_min and year_max else (None, None)

filtered_df = df.copy()
if selected_make:
    filtered_df = filtered_df[filtered_df["make"].isin(selected_make)]
if selected_years[0] and selected_years[1]:
    filtered_df = filtered_df[(filtered_df["year"] >= selected_years[0]) & (filtered_df["year"] <= selected_years[1])]

st.title("Airline Accidents Dashboard")
st.write("### Dataset Preview")
st.dataframe(filtered_df.head())

# -------------------------------
# Step 5: Accidents per Year
# -------------------------------
if "year" in filtered_df.columns:
    st.subheader("Accidents per Year")
    accidents_by_year = filtered_df.groupby("year").size()

    if not accidents_by_year.empty:
        fig, ax = plt.subplots(figsize=(10,5))
        accidents_by_year.plot(kind="bar", ax=ax)
        ax.set_title("Number of Accidents per Year")
        ax.set_ylabel("Accidents")
        ax.set_xlabel("Year")
        st.pyplot(fig)
    else:
        st.warning("No accident data for selected filters.")

# -------------------------------
# Step 6: Fatalities Trend
# -------------------------------
if "year" in filtered_df.columns and "total_fatal_injuries" in filtered_df.columns:
    st.subheader("Fatalities Trend Over Time")

    fatalities_trend = filtered_df.groupby("year")["total_fatal_injuries"].sum().sort_index()

    if not fatalities_trend.empty and fatalities_trend.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        fatalities_trend.plot(kind="line", marker="o", ax=ax)
        ax.set_title("Fatalities per Year")
        ax.set_ylabel("Fatalities")
        ax.set_xlabel("Year")
        st.pyplot(fig)
    else:
        st.warning("No fatality data available for selected filters.")

# -------------------------------
# Step 7: Accident Severity Over Time
# -------------------------------
injury_cols = ["total_fatal_injuries","total_serious_injuries","total_minor_injuries","total_uninjured"]
injury_cols = [col for col in injury_cols if col in filtered_df.columns]

if all(col in filtered_df.columns for col in injury_cols) and "year" in filtered_df.columns:
    st.subheader("Accident Severity Over Time")

    # Correct: first numeric, then groupby sum
    severity_trend = filtered_df.groupby("year")[injury_cols].sum()

    if not severity_trend.empty:
        fig, ax = plt.subplots(figsize=(12,6))
        severity_trend.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Accident Severity Distribution Over Time")
        ax.set_ylabel("Number of People")
        ax.set_xlabel("Year")
        st.pyplot(fig)
    else:
        st.warning("No severity data available for selected filters.")

# -------------------------------
# Step 8: Severity by Aircraft Make
# -------------------------------
if all(col in filtered_df.columns for col in injury_cols) and "make" in filtered_df.columns:
    st.subheader("Accident Severity by Aircraft Make")

    severity_by_make = filtered_df.groupby("make")[injury_cols].sum()
    severity_by_make = severity_by_make.sort_values(by="total_fatal_injuries", ascending=False).head(10)

    if not severity_by_make.empty:
        fig, ax = plt.subplots(figsize=(12,6))
        severity_by_make.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Top 10 Aircraft Makes by Accident Severity")
        ax.set_ylabel("Number of People")
        ax.set_xlabel("Aircraft Make")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No severity data available by aircraft make.")

# -------------------------------
# Step 9: Accidents by Airline
# -------------------------------
if "airline" in filtered_df.columns:
    st.subheader("Accidents by Airline")

    accidents_by_airline = filtered_df["airline"].value_counts().head(10)

    if not accidents_by_airline.empty:
        fig, ax = plt.subplots(figsize=(12,6))
        accidents_by_airline.plot(kind="bar", ax=ax, color="teal")
        ax.set_title("Top 10 Airlines by Number of Accidents")
        ax.set_ylabel("Number of Accidents")
        ax.set_xlabel("Airline")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No airline accident data available for selected filters.")

# -------------------------------
# Step 10: Fatalities by Airline
# -------------------------------
if "airline" in filtered_df.columns and "total_fatal_injuries" in filtered_df.columns:
    st.subheader("Fatalities by Airline")

    fatalities_by_airline = filtered_df.groupby("airline")["total_fatal_injuries"].sum().sort_values(ascending=False).head(10)

    if not fatalities_by_airline.empty:
        fig, ax = plt.subplots(figsize=(12,6))
        fatalities_by_airline.plot(kind="bar", ax=ax, color="crimson")
        ax.set_title("Top 10 Airlines by Fatalities")
        ax.set_ylabel("Fatalities")
        ax.set_xlabel("Airline")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No fatality data available for airlines.")

# -------------------------------
# Step 11: Correlation Heatmap
# -------------------------------
numeric_cols = injury_cols

if numeric_cols:
    st.subheader("Correlation Heatmap")

    corr_matrix = filtered_df[numeric_cols].corr()

    if not corr_matrix.empty:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation between Injury Columns")
        st.pyplot(fig)
    else:
        st.warning("No numeric data available for correlation.")
