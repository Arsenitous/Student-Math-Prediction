import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Set the page configuration
st.set_page_config(
    page_title="Student Math Score Prediction", page_icon="🎓", layout="centered"
)

st.title("🎓 Student Math Score Prediction App")

st.markdown(
    """
Welcome to the **Student Math Score Prediction** tool.  
This application uses a Machine Learning model to **predict a student's math exam score** 
based on their demographic and academic background.

### 📋 How it Works
Fill in the student's profile using the form below, then click **Predict Score** to get 
the estimated math score. The model was trained on a dataset of 1,000 students.
"""
)

st.divider()

# --- Input Form ---
st.subheader("📝 Student Profile Input")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender", options=["female", "male"], help="Select the student's gender."
        )

        race_ethnicity = st.selectbox(
            "Race / Ethnicity",
            options=["group A", "group B", "group C", "group D", "group E"],
            help="Select the student's race/ethnicity group.",
        )

        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
            options=[
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree",
            ],
            help="Select the highest education level of the student's parent(s).",
        )

    with col2:
        lunch = st.selectbox(
            "Lunch Type",
            options=["standard", "free/reduced"],
            help="Does the student have a standard or free/reduced lunch plan?",
        )

        test_preparation_course = st.selectbox(
            "Test Preparation Course",
            options=["none", "completed"],
            help="Did the student complete a test preparation course?",
        )

        reading_score = st.number_input(
            "Reading Score (0–100)",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
            help="Enter the student's reading exam score.",
        )

        writing_score = st.number_input(
            "Writing Score (0–100)",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
            help="Enter the student's writing exam score.",
        )

    submitted = st.form_submit_button(
        "🔮 Predict Math Score", type="primary", use_container_width=True
    )

# --- Prediction Logic ---
if submitted:
    with st.spinner("Running prediction model..."):
        try:
            # Build the input data object
            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score,
            )

            pred_df = data.get_data_as_data_frame()

            # Run prediction pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            predicted_score = round(float(results[0]), 2)

            st.success("✅ Prediction successful!")
            st.divider()

            # --- Result Display ---
            st.subheader("🎯 Prediction Result")

            # Determine performance grade
            if predicted_score >= 80:
                grade = "🏆 Excellent"
            elif predicted_score >= 65:
                grade = "✅ Good"
            elif predicted_score >= 50:
                grade = "⚠️ Average"
            else:
                grade = "❌ Needs Improvement"

            col1, col2 = st.columns(2)
            col1.metric("📊 Predicted Math Score", f"{predicted_score} / 100")
            col2.metric("📈 Performance Grade", grade)

            # Progress bar
            st.progress(int(min(predicted_score, 100)) / 100)

            st.info(
                "**Grade Scale:** 🏆 Excellent (≥80) | ✅ Good (65–79) | ⚠️ Average (50–64) | ❌ Needs Improvement (<50)"
            )

            # --- Input Summary ---
            with st.expander("📋 View Full Input Summary"):
                summary_df = pd.DataFrame(
                    [
                        {
                            "Gender": gender,
                            "Race/Ethnicity": race_ethnicity,
                            "Parental Education": parental_level_of_education,
                            "Lunch": lunch,
                            "Test Preparation": test_preparation_course,
                            "Reading Score": reading_score,
                            "Writing Score": writing_score,
                            "Predicted Math Score": predicted_score,
                        }
                    ]
                )
                st.dataframe(summary_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")

st.divider()
st.caption(
    "🤖 Model: AdaBoost Regressor | Dataset: Students Performance in Exams | Built with Streamlit"
)
