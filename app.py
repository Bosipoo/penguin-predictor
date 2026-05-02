"""
Penguin Body Mass Predictor
A Streamlit app that predicts a penguin's body mass from its measurements.

This is the reference app for Week 11 - Day 2.
We build this together in class, then deploy it to Streamlit Cloud.
"""

import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
# st.set_page_config controls things like the browser tab title and the layout.
# It MUST be the first Streamlit command in the script.
st.set_page_config(
    page_title="Penguin Predictor",
    page_icon="🐧",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Load the trained pipeline
# ---------------------------------------------------------------------------
# Remember: Streamlit re-runs this whole script every time the user clicks
# anything. Without caching, we'd reload the model file on every click — slow!
# @st.cache_resource tells Streamlit "load this once, then reuse it".
@st.cache_resource
def load_model():
    """Load the trained pipeline from disk."""
    return joblib.load("model/penguin_model.joblib")


pipeline = load_model()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🐧 Penguin Body Mass Predictor")
st.write(
    "Enter a penguin's measurements below and we'll predict its body mass. "
    "Built on the Palmer Penguins dataset using a Random Forest pipeline."
)

st.divider()


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
# We organise the inputs in two columns so the form feels less like a long list.
st.subheader("Penguin details")

col1, col2 = st.columns(2)

with col1:
    species = st.selectbox(
        "Species",
        options=["Adelie", "Chinstrap", "Gentoo"],
        help="Which species of penguin?",
    )

    island = st.selectbox(
        "Island",
        options=["Biscoe", "Dream", "Torgersen"],
        help="Which island was the penguin observed on?",
    )

    sex = st.radio(
        "Sex",
        options=["Male", "Female"],
        horizontal=True,
    )

with col2:
    bill_length_mm = st.number_input(
        "Bill length (mm)",
        min_value=30.0,
        max_value=60.0,
        value=44.0,
        step=0.1,
    )

    bill_depth_mm = st.number_input(
        "Bill depth (mm)",
        min_value=13.0,
        max_value=22.0,
        value=17.0,
        step=0.1,
    )

    flipper_length_mm = st.number_input(
        "Flipper length (mm)",
        min_value=170.0,
        max_value=235.0,
        value=200.0,
        step=1.0,
    )


# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
st.divider()

if st.button("Predict body mass", type="primary", use_container_width=True):

    # Build a one-row DataFrame from the user inputs.
    # The column names MUST match the names the pipeline was trained on.
    user_input = pd.DataFrame([{
        "species": species,
        "island": island,
        "sex": sex,
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
    }])

    # Pass the raw input to the pipeline. The pipeline handles encoding
    # and scaling internally, then runs the model.
    prediction = pipeline.predict(user_input)[0]

    # Show the result in a friendly format
    st.success(f"### Predicted body mass: **{prediction:,.0f} grams**")
    st.caption(f"That's about {prediction / 1000:.2f} kg — roughly the weight of a small bag of rice.")

    # Show what the user typed in (helps with trust)
    with st.expander("See the values used for this prediction"):
        st.dataframe(user_input, hide_index=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Built with Streamlit · Trained on the Palmer Penguins dataset · "
    "Week 11 Data Science Cohort"
)