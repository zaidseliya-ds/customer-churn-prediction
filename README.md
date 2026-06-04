# Telecom Customer Churn Prediction & Dashboard

This is my Semester VI Data Science and Machine Learning project. It is designed to predict telecom customer churn with 94% accuracy using Gradient Boosting (XGBoost logic) and visualizes the model's decisions using SHAP explainability values.

## Key Features
* **XGBoost Classifier:** Trains on telecom customer records to predict who is likely to leave.
* **SHAP Plots:** Displays clear charts showing factors that impact the model's prediction.
* **Interactive Dashboard:** Built with Streamlit so users can check live churn risk using sliders.

## Project Structure
```text
Customer_Churn_Prediction_Dashboard/
│
├── src/
│   ├── model.py         # Data preprocessing and model training script
│   └── app.py           # Streamlit dashboard and SHAP visualization app
│
├── .gitignore           # Ignores cache and python notebook files
├── requirements.txt     # Python dependencies needed to run the project
└── README.md            # Project documentation (This file)

Install Dependencies: pip install -r requirements.txt
Train the Machine Learning Model: python src/model.py
Start the Web Dashboard: streamlit run src/app.py

## Tech Stack Used
Python 3, Scikit-Learn, Joblib, NumPy, Pandas, Streamlit, Plotly Express
