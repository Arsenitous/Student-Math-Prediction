# 🎓 Student Math Score Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B)](https://ml-project-efj6w5gi4ybs7tmfbyh5kx.streamlit.app/)
[![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey)](https://flask.palletsprojects.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)](https://scikit-learn.org)

### 🚀 Try the Live Demo!

> 🔗 **[https://ml-project-efj6w5gi4ybs7tmfbyh5kx.streamlit.app/](https://ml-project-efj6w5gi4ybs7tmfbyh5kx.streamlit.app/)**

---

## 📚 Project Overview

This project implements an **end-to-end Machine Learning pipeline** to predict a student's **math exam score** based on demographic and academic background features.

The goal is to explore how factors such as gender, parental education, lunch type, and test preparation affect student academic performance. The project includes both a **Flask web app** and an **interactive Streamlit app** for making predictions.

### 🎯 Key Features
- ✅ **End-to-End ML Pipeline** — Data ingestion → Preprocessing → Model Training → Evaluation
- ✅ **AdaBoost Regressor** — Ensemble boosting model for accurate score prediction
- ✅ **Interactive Streamlit App** — Predict math scores with a simple web form
- ✅ **Flask Backend** — REST-ready web app with HTML form interface
- ✅ **Modular Code Structure** — Clean separation of components, pipelines, and utilities

---

## 🌐 Option A: Streamlit Web App (Recommended)

The Streamlit app provides a clean, interactive form to input a student's profile and instantly receive a **predicted math score** with a performance grade.

### 🌍 Live App

The app is already deployed and publicly accessible at:

> 🔗 **[https://ml-project-efj6w5gi4ybs7tmfbyh5kx.streamlit.app/](https://ml-project-efj6w5gi4ybs7tmfbyh5kx.streamlit.app/)**

### Run Locally

```bash
streamlit run deploy.py
```

Then open your browser and navigate to `http://localhost:8501`.

### How to Use the App

1. **Fill in the Student Profile** — select gender, race/ethnicity, parental education, lunch type, and test prep status.
2. **Enter Reading & Writing Scores** — input the student's scores for the other two subjects.
3. **Click Predict Math Score** — the model instantly returns a predicted math score.
4. **View the Result** — see the numeric score, a performance grade badge, and a progress bar.

### Input Features

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Student's gender: `female` or `male` |
| `race_ethnicity` | Categorical | Ethnic group: `group A` to `group E` |
| `parental_level_of_education` | Categorical | Parent's highest education level |
| `lunch` | Categorical | Lunch plan: `standard` or `free/reduced` |
| `test_preparation_course` | Categorical | Completed test prep: `none` or `completed` |
| `reading_score` | Numerical | Student's reading exam score (0–100) |
| `writing_score` | Numerical | Student's writing exam score (0–100) |

### Output

| Grade | Score Range | Label |
|---|---|---|
| 🏆 Excellent | ≥ 80 | Top performance |
| ✅ Good | 65–79 | Above average |
| ⚠️ Average | 50–64 | Meets expectations |
| ❌ Needs Improvement | < 50 | Below expectations |

---

## ⚙️ Option B: Flask Web App (For Developers)

The original Flask app provides a form-based web interface for predictions.

### Run Locally

```bash
python app.py
```

The server starts at `http://localhost:5000`. Navigate to `/predictdata` to access the prediction form.

---

## 🚀 Getting Started (Local Setup)

### Prerequisites
- Python 3.8+

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Arsenitous/ML-Project.git
cd ML-Project
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

### 2. Run the Training Pipeline (Optional)

If you want to retrain the model from scratch:

```bash
python src/components/data_ingestion.py
```

This will:
1. Ingest raw data from `artifacts/raw.csv`
2. Split into train/test sets
3. Apply preprocessing (encoding + scaling)
4. Train the model and save `artifacts/model.pkl` and `artifacts/preprocessor.pkl`

### 3. Launch the App

```bash
# Streamlit (recommended)
streamlit run deploy.py

# or Flask
python app.py
```

---

## 🗂️ Project Structure

```
Student Prediction/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Load and split raw data
│   │   ├── data_transformation.py  # Feature encoding & scaling
│   │   └── model_trainer.py        # Train and evaluate the model
│   ├── pipeline/
│   │   ├── predict_pipeline.py     # PredictPipeline & CustomData classes
│   │   └── train_pipeline.py       # End-to-end training orchestrator
│   ├── exception.py                # Custom exception handler
│   ├── logger.py                   # Logging configuration
│   └── utils.py                    # Utility functions (load_object, etc.)
├── artifacts/
│   ├── model.pkl                   # Trained AdaBoost model
│   ├── preprocessor.pkl            # Fitted preprocessing pipeline
│   ├── raw.csv                     # Raw dataset (~1,000 students)
│   ├── train.csv                   # Training split
│   └── test.csv                    # Test split
├── notebook/                       # Exploratory Data Analysis & experiments
├── templates/                      # HTML templates for Flask app
├── app.py                          # Flask web app entry point
├── deploy.py                       # Streamlit web app entry point
├── requirements.txt                # Python dependencies
└── README.md                       # You are here
```

---

## 📦 Dependencies

| Category | Libraries |
|---|---|
| **Web App (Streamlit)** | `streamlit` |
| **Web App (Flask)** | `flask` |
| **ML & Data** | `scikit-learn`, `pandas`, `numpy`, `catboost` |
| **Utilities** | `dill` |

---

## 📊 Model & Dataset

| Item | Details |
|---|---|
| **Dataset** | [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) |
| **Samples** | ~1,000 students |
| **Features** | 7 input features (5 categorical, 2 numerical) |
| **Target** | `math_score` (continuous, 0–100) |
| **Model** | AdaBoost Regressor |
| **Preprocessing** | OneHotEncoder (categorical) + StandardScaler (numerical) |

---

*Happy learning! 🎉 Built with ❤️ for ML education.*
