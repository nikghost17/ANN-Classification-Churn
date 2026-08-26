# 🧠 Customer Churn Prediction — ANN Classification

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Keras-ANN-red?style=for-the-badge&logo=keras" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/scikit--learn-Preprocessing-blue?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python" />
</p>

---

## 📌 Overview

A binary classification model built with an **Artificial Neural Network (ANN)** to predict whether a bank customer will churn (leave the bank). The project covers the complete ML workflow — from raw CSV data to a deployed interactive web application.

> **Dataset**: [Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) — 10,000 bank customers with 14 features.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔢 **End-to-End ML Pipeline** | Data loading → preprocessing → training → evaluation → deployment |
| 🧠 **ANN with Keras** | 2-hidden-layer Sequential model with ReLU activations and Sigmoid output |
| 📉 **EarlyStopping + TensorBoard** | Prevents overfitting; full training visualization via TensorBoard |
| 🔄 **Serialized Artifacts** | Encoders and scaler saved as `.pkl` for reproducible inference |
| 🌐 **Streamlit Web App** | Interactive UI — input customer details, get real-time churn probability |

---

## 🏗️ Model Architecture

```
Input Layer  →  12 features
                (CreditScore, Gender, Age, Tenure, Balance,
                 NumOfProducts, HasCrCard, IsActiveMember,
                 EstimatedSalary, Geography_France,
                 Geography_Germany, Geography_Spain)
    ↓
Dense(64, activation='relu')      # Hidden Layer 1
    ↓
Dense(32, activation='relu')      # Hidden Layer 2
    ↓
Dense(1,  activation='sigmoid')   # Output — Churn Probability
```

| Property | Value |
|---|---|
| **Optimizer** | Adam (lr = 0.01) |
| **Loss** | Binary Crossentropy |
| **Callbacks** | EarlyStopping (patience=10), TensorBoard |
| **Train/Test Split** | 80% / 20% |
| **Best Validation Accuracy** | ~86.95% |
| **Total Parameters** | 2,945 |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | TensorFlow 2.15 / Keras |
| **Preprocessing** | scikit-learn (StandardScaler, LabelEncoder, OneHotEncoder) |
| **Data** | pandas, NumPy |
| **Visualization** | TensorBoard, matplotlib |
| **Web App** | Streamlit |
| **Serialization** | pickle |

---

## 📂 Project Structure

```
ANN-Classification-Churn/
│
├── Churn_Modelling.csv          # Raw dataset (10,000 rows × 14 columns)
├── experiments.ipynb            # Full pipeline: EDA → preprocessing → training
├── prediction.ipynb             # Inference notebook using saved model
├── app.py                       # Streamlit web application
│
├── model.h5                     # Trained Keras ANN model
├── scaler.pkl                   # Fitted StandardScaler
├── label_encoder_gender.pkl     # Fitted LabelEncoder (Gender)
├── one_hot_encoder_geo.pkl      # Fitted OneHotEncoder (Geography)
│
├── requirements.txt             # Python dependencies
└── runtime.txt                  # Python runtime specification
```

---

## 🚀 Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ANN-Classification-Churn.git
cd ANN-Classification-Churn
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📓 Notebooks

### `experiments.ipynb` — Full ML Pipeline

Covers:
1. **Data Loading** — Load `Churn_Modelling.csv`, inspect with `data.head()`
2. **Preprocessing**
   - Drop irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
   - `LabelEncoder` for `Gender` (Male → 1, Female → 0)
   - `OneHotEncoder` for `Geography` (France / Germany / Spain → 3 binary columns)
   - `StandardScaler` to normalize all features
3. **Train / Test Split** — 80% train, 20% test, `random_state=42`
4. **ANN Training** — with EarlyStopping and TensorBoard callbacks
5. **Model Saving** — `model.h5` + serialized encoders/scaler as `.pkl`

### `prediction.ipynb` — Inference

Loads the saved model and artifacts, runs a sample customer through the full preprocessing pipeline, and outputs the churn probability.

---

## 🌐 Streamlit Web App

The app (`app.py`) provides an interactive form where you can:
- Select customer **Geography** and **Gender**
- Adjust **Age**, **Tenure**, **Number of Products** via sliders
- Enter **Balance**, **Credit Score**, **Estimated Salary**
- Toggle **Has Credit Card** and **Is Active Member**

On submission, the app:
1. Applies the same preprocessing pipeline (encode → scale)
2. Passes scaled input through the trained ANN
3. Displays the **churn probability** and a plain-language verdict

---

## 📊 Results

| Metric | Value |
|---|---|
| Validation Accuracy | ~86.95% |
| Training Epochs (early stopped) | 17 |
| Model Size | ~66 KB |

> Training stopped early at Epoch 17 (patience=10) as validation loss stopped improving, preventing overfitting.

---

## 🔁 Training Visualization

Launch TensorBoard to view loss/accuracy curves:

```bash
tensorboard --logdir logs/fit
```

Open **http://localhost:6006** in your browser.

---

## 📋 Dataset Description

| Column | Type | Description |
|---|---|---|
| `CreditScore` | int | Customer credit score (350–850) |
| `Geography` | categorical | Country: France, Germany, or Spain |
| `Gender` | categorical | Male or Female |
| `Age` | int | Customer age (18–92) |
| `Tenure` | int | Years with the bank (0–10) |
| `Balance` | float | Account balance |
| `NumOfProducts` | int | Number of bank products (1–4) |
| `HasCrCard` | binary | Has a credit card (0/1) |
| `IsActiveMember` | binary | Is an active member (0/1) |
| `EstimatedSalary` | float | Estimated annual salary |
| `Exited` | binary | **Target** — 1 = churned, 0 = stayed |

---

## 🔮 Future Improvements

- [ ] Add confusion matrix, AUC-ROC, precision/recall evaluation
- [ ] Add EDA charts (class distribution, correlation heatmap)
- [ ] Hyperparameter tuning (Keras Tuner / grid search)
- [ ] Add Dropout and Batch Normalization layers
- [ ] Migrate model save format from `.h5` to `.keras`
- [ ] Class imbalance handling (SMOTE or class weights)

---

## 📄 License

MIT — feel free to use, modify, and distribute.

---

<p align="center">
  Built by <a href="https://github.com/nikghost17">Nikhil Nilesh Vedak</a>
</p>
