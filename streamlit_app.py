import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Aviation Accident Analysis Dashboard")

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("airline_accidents.csv", low_memory=False)

st.subheader("Dataset Preview")
st.write("Columns in dataset:", df.columns.tolist())
st.write(df.head())

# -------------------------------
# Clean and prepare date column
# -------------------------------
if "Event Date" in df.columns:
    df["Event Date"] = pd.to_datetime(df["Event Date"], errors="coerce")
    df["Year"] = df["Event Date"].dt.year
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
else:
    st.error("The dataset does not contain 'Event Date'. Please check column names above.")

# -------------------------------
# Convert injury columns to numeric
# -------------------------------
injury_cols = [
    "Total Fatal Injuries",
    "Total Serious Injuries",
    "Total Minor Injuries",
    "Total Uninjured"
]

for col in injury_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# -------------------------------
# Step 5: Add filters
# -------------------------------
st.header("Filter Accidents")

years = sorted(df['Year'].dropna().unique())
selected_years = st.multiselect("Select Year(s):", years, default=years)

makes = sorted(df['Make'].dropna().unique())
selected_makes = st.multiselect("Select Aircraft Make(s):", makes, default=makes[:5])

# Apply filters
filtered_df = df[(df['Year'].isin(selected_years)) & (df['Make'].isin(selected_makes))]

st.write(f"### Showing {len(filtered_df)} records after filtering")
st.dataframe(filtered_df.head(50))

# -------------------------------
# Fatalities trend over time
# -------------------------------
if "Total Fatal Injuries" in filtered_df.columns and "Year" in filtered_df.columns:
    fatalities_trend = filtered_df.groupby("Year")["Total Fatal Injuries"].sum()

    if fatalities_trend.empty:
        st.warning("No fatalities data available for selected filters.")
    else:
        st.subheader("Fatalities Trend Over Time")
        fig, ax = plt.subplots(figsize=(10,5))
        fatalities_trend.plot(kind="line", marker="o", ax=ax)
        ax.set_title("Fatalities Trend Over Time")
        ax.set_ylabel("Total Fatalities")
        ax.set_xlabel("Year")
        st.pyplot(fig)

# -------------------------------
# Top 10 Aircraft Makes by Number of Accidents
# -------------------------------
if "Make" in filtered_df.columns:
    accidents_by_make = filtered_df["Make"].value_counts().head(10)

    st.subheader("Top 10 Aircraft Makes by Number of Accidents")
    fig, ax = plt.subplots(figsize=(10,5))
    accidents_by_make.plot(kind="bar", ax=ax)
    ax.set_title("Top 10 Aircraft Makes by Accidents")
    ax.set_ylabel("Number of Accidents")
    ax.set_xlabel("Aircraft Make")
    st.pyplot(fig)

# -------------------------------
# Fatalities by Aircraft Make
# -------------------------------
if "Make" in filtered_df.columns and "Total Fatal Injuries" in filtered_df.columns:
    fatalities_by_make = (
        filtered_df.groupby("Make")["Total Fatal Injuries"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.subheader("Top 10 Aircraft Makes by Fatalities")
    fig, ax = plt.subplots(figsize=(10,5))
    fatalities_by_make.plot(kind="bar", ax=ax)
    ax.set_title("Top 10 Aircraft Makes by Fatalities")
    ax.set_ylabel("Total Fatalities")
    ax.set_xlabel("Aircraft Make")
    st.pyplot(fig)

# -------------------------------
# Distribution of Injuries
# -------------------------------
if all(col in filtered_df.columns for col in injury_cols):
    injury_distribution = filtered_df[injury_cols].sum()

    st.subheader("Distribution of Injuries in Aviation Accidents")
    fig, ax = plt.subplots(figsize=(7,7))
    injury_distribution.plot(
        kind="pie", autopct="%1.1f%%", startangle=140, ax=ax
    )
    ax.set_ylabel("")
    ax.set_title("Distribution of Injuries")
    st.pyplot(fig)

# -------------------------------
# Step 6: Top 10 Locations by Number of Accidents
# -------------------------------
if "Location" in filtered_df.columns:
    top_locations = filtered_df["Location"].value_counts().head(10)

    st.subheader("Top 10 Locations by Number of Accidents")
    fig, ax = plt.subplots(figsize=(10,5))
    top_locations.plot(kind="bar", ax=ax)
    ax.set_title("Top 10 Locations by Number of Accidents")
    ax.set_ylabel("Number of Accidents")
    ax.set_xlabel("Location")
    st.pyplot(fig)

# -------------------------------
# Wrap up
# -------------------------------
st.success("Dashboard loaded successfully with filters applied!")
