# Loan Approval Prediction – End-to-End ML Pipeline

This project builds a **complete credit-risk prediction system** for **FinTrust**, a digital lending company, using a simple **three-script structure**:

1. **Data Exploration & Analysis** – `model.py`
2. **Data Preprocessing** – `data_cleaner.py`
3. **Model Training & Evaluation** – `main.py`

The final output is a **Logistic Regression model** capable of predicting whether a loan application will be **Approved** or **Rejected**.

---

## 📂 Project Structure

```
loan-approval/
│
├── loan_data.csv              # Raw dataset
├── model.py                   # Part 1: Data exploration & EDA
├── data_cleaner.py            # Part 2: Data cleaning & preprocessing
├── main.py                    # Part 3: Logistic Regression training & evaluation
└── README.md
```

---

## 1️⃣ model.py – Data Exploration and Analysis

Goal: **Understand the dataset before modeling**

Key actions inside `model.py`:

* Load dataset with `pandas.read_csv`
* Display shape and column names
* Check data types to separate numerical & categorical features
* Identify missing values
* Generate summary statistics using `.describe()`
* Perform simple visualizations (histograms, boxplots, correlations)

Insights:

* Found missing values in some columns (e.g., `Gender`, `LoanAmount`).
* Observed class imbalance in `Loan_Status`.

---

## 2️⃣ data\_cleaner.py – Preprocessing

Goal: **Clean and prepare data for modeling**

Main steps:

* **Handle Missing Values**

  * Numerical columns: imputed with mean/median
  * Categorical columns: filled with mode
* **Encode Categorical Variables** using `LabelEncoder`
* **Scale Numerical Features** using `MinMaxScaler`
* Output a cleaned, ready-to-train dataset and save transformation tools (`encoder.pkl`, `scaler.pkl`).

---

## 3️⃣ main.py – Model Training & Evaluation

Goal: **Train and evaluate the predictive model**

Operations in `main.py`:

* Load the preprocessed dataset from `data_cleaner.py`
* Train a `sklearn.linear_model.LogisticRegression` model
* Evaluate with:

  * Accuracy
  * Confusion Matrix
  * Precision, Recall, F1-Score (optional)
* Save the trained model as `logistic_model.pkl` for future predictions.

---

## 🚀 Quick Start

### Prerequisites

* Python 3.9+
* Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Full Pipeline

1. **Explore Data**

   ```bash
   python model.py
   ```

2. **Clean & Preprocess Data**

   ```bash
   python data_cleaner.py
   ```

3. **Train & Evaluate Model**

   ```bash
   python main.py
   ```

This produces:

* `encoder.pkl` and `scaler.pkl` for consistent transformations
* `logistic_model.pkl` trained model file

---

## 🧩 Example Prediction

```python
import joblib
import pandas as pd

# Load saved objects
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
model  = joblib.load("logistic_model.pkl")

# New applicant data as DataFrame
new_applicant = pd.DataFrame({
    "Gender": ["Male"],
    "Married": ["Yes"],
    "ApplicantIncome": [5000],
    "LoanAmount": [150],
    # ... other features ...
})

# Apply preprocessing (encoding & scaling) before prediction
new_applicant_encoded = encoder.transform(new_applicant)
new_applicant_scaled  = scaler.transform(new_applicant_encoded)

# Predict
prediction = model.predict(new_applicant_scaled)
print("Loan Status:", "Approved" if prediction[0] == 1 else "Rejected")
```

---

## 📈 Future Improvements

* Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
* Handle class imbalance using SMOTE or class weights
* Experiment with other classifiers such as Random Forest or XGBoost

---

## 🛠️ Tech Stack

* **Python 3**
* **Pandas, NumPy, Matplotlib, Seaborn**
* **Scikit-learn**

---

## ✍️ Author

**divs-spec**
Contributions & pull requests are welcome!
