# 🧠 ANN Customer Churn Prediction — Complete Project Deep-Dive

> This document explains **every internal working** of this project — the data, preprocessing pipeline, model architecture, training strategy, inference flow, and Streamlit app — at a level of depth suitable for a technical interview or for an LLM to fully understand the project.

---

## 1. The 30-Second Elevator Pitch

> "I built a binary classification model using an Artificial Neural Network in TensorFlow/Keras to predict whether a bank customer will churn. The project covers the full ML lifecycle: raw CSV ingestion, categorical encoding, feature scaling, ANN training with EarlyStopping and TensorBoard, serialization of all preprocessing artifacts, and deployment as an interactive Streamlit web app."

**Key points to emphasize:**
- **Not just model training** — the complete pipeline from raw data to a live, usable web app
- **Proper ML hygiene** — train/test split, feature scaling applied correctly (fit on train, transform on test), artifacts serialized for reproducible inference
- **Production-ready inference** — the Streamlit app loads the saved model and encoders and runs the exact same preprocessing pipeline as training

---

## 2. The Problem

**Domain**: Banking / Customer Retention  
**Task**: Binary classification — predict `Exited` (1 = churned, 0 = stayed)  
**Why it matters**: Acquiring a new customer costs 5–7x more than retaining one. Predicting churn lets banks intervene proactively with at-risk customers.

---

## 3. The Dataset

**File**: `Churn_Modelling.csv`  
**Size**: 10,000 rows × 14 columns  
**Source**: [Kaggle — Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

### All Columns

| Column | Type | Range / Values | Used? |
|---|---|---|---|
| `RowNumber` | int | 1–10000 | ❌ Dropped |
| `CustomerId` | int | unique IDs | ❌ Dropped |
| `Surname` | string | names | ❌ Dropped |
| `CreditScore` | int | 350–850 | ✅ |
| `Geography` | categorical | France, Germany, Spain | ✅ OneHotEncoded |
| `Gender` | categorical | Male, Female | ✅ LabelEncoded |
| `Age` | int | 18–92 | ✅ |
| `Tenure` | int | 0–10 | ✅ |
| `Balance` | float | 0 – 250,000+ | ✅ |
| `NumOfProducts` | int | 1–4 | ✅ |
| `HasCrCard` | binary | 0 or 1 | ✅ |
| `IsActiveMember` | binary | 0 or 1 | ✅ |
| `EstimatedSalary` | float | 11–200,000+ | ✅ |
| `Exited` | binary | 0 or 1 | ✅ TARGET |

**Class balance**: ~20% churned (Exited=1), ~80% stayed (Exited=0) — mildly imbalanced.

---

## 4. Preprocessing Pipeline (Step by Step)

### Step 1 — Drop irrelevant columns
```python
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
```
`RowNumber` and `CustomerId` are identifiers with no predictive power.
`Surname` is a high-cardinality string with no meaningful signal.

### Step 2 — Label Encode `Gender`
```python
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
# Female → 0, Male → 1
```
Binary categorical → single integer column. LabelEncoder is appropriate here because there are only 2 classes (no ordinal relationship implied for a binary variable).

### Step 3 — OneHot Encode `Geography`
```python
one_hot_encoder_geo = OneHotEncoder()
geo_encoder = one_hot_encoder_geo.fit_transform(data[['Geography']])
geo_encoder_df = pd.DataFrame(geo_encoder.toarray(),
    columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
# Creates: Geography_France, Geography_Germany, Geography_Spain
data = pd.concat([data.drop('Geography', axis=1), geo_encoder_df], axis=1)
```
3-class categorical → 3 binary columns. OneHotEncoder is used (not LabelEncoder) because Label Encoding would imply France < Germany < Spain, which is a false ordinal relationship.

### Step 4 — Save encoders as pickle files
```python
with open('label_encoder_gender.pkl', 'wb') as file:
    pickle.dump(label_encoder_gender, file)
with open('one_hot_encoder_geo.pkl', 'wb') as file:
    pickle.dump(one_hot_encoder_geo, file)
```
Critical for inference: the app must use the **same fitted encoders** as training, not re-fit on new data.

### Step 5 — Split features and target
```python
X = data.drop('Exited', axis=1)   # shape: (10000, 12)
y = data['Exited']                 # shape: (10000,)
```

### Step 6 — Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# X_train: 8000 rows, X_test: 2000 rows
```
`random_state=42` ensures reproducibility.

### Step 7 — Standard Scaling
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit + transform on training data
X_test  = scaler.transform(X_test)        # ONLY transform on test data
```
**Critical detail**: `fit_transform` on train, `transform` only on test. Fitting on the test set would cause **data leakage** — the model would indirectly see test set statistics during training.

StandardScaler converts each feature to mean=0, std=1. This is required for ANNs because gradient descent is sensitive to feature scale — without scaling, features like `Balance` (0–250,000) would dominate features like `HasCrCard` (0 or 1).

### Step 8 — Save scaler
```python
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
```
The Streamlit app loads this exact scaler to normalize user input at inference time.

---

## 5. Final Feature Set (12 Input Features)

After preprocessing, `X` has 12 columns:

| # | Feature | Original Column | Encoding |
|---|---|---|---|
| 1 | CreditScore | CreditScore | Scaled |
| 2 | Gender | Gender | LabelEncoded → 0/1, Scaled |
| 3 | Age | Age | Scaled |
| 4 | Tenure | Tenure | Scaled |
| 5 | Balance | Balance | Scaled |
| 6 | NumOfProducts | NumOfProducts | Scaled |
| 7 | HasCrCard | HasCrCard | Scaled |
| 8 | IsActiveMember | IsActiveMember | Scaled |
| 9 | EstimatedSalary | EstimatedSalary | Scaled |
| 10 | Geography_France | Geography | OneHotEncoded, Scaled |
| 11 | Geography_Germany | Geography | OneHotEncoded, Scaled |
| 12 | Geography_Spain | Geography | OneHotEncoded, Scaled |

---

## 6. ANN Model Architecture

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(12,)),  # Hidden Layer 1
    Dense(32, activation='relu'),                      # Hidden Layer 2
    Dense(1,  activation='sigmoid')                    # Output Layer
])
```

### Why These Choices?

| Decision | Reason |
|---|---|
| **ReLU activation** in hidden layers | Avoids vanishing gradient problem; fast to compute; standard for classification hidden layers |
| **Sigmoid output** | Maps output to (0, 1) — interpreted directly as churn probability |
| **64 → 32 neurons** | Funnel architecture: wider early layers capture feature combinations, narrower later layers distill to the prediction |
| **2 hidden layers** | Enough capacity to learn non-linear feature interactions; more layers would risk overfitting on 10k rows |
| **Binary Crossentropy loss** | Standard loss for binary classification; measures log-likelihood of the true class |
| **Adam optimizer (lr=0.01)** | Adaptive learning rate — combines momentum and RMSProp; generally outperforms plain SGD |

### Model Summary
```
Layer (type)         Output Shape    Param #
──────────────────────────────────────────────
dense (Dense)        (None, 64)      832       ← 12×64 weights + 64 biases
dense_1 (Dense)      (None, 32)      2,080     ← 64×32 weights + 32 biases
dense_2 (Dense)      (None, 1)       33        ← 32×1 weights + 1 bias
──────────────────────────────────────────────
Total params: 2,945 (11.50 KB)
```

---

## 7. Training Strategy

```python
# Compile
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[tensorboard_callback, early_stopping_callback]
)
```

### EarlyStopping Explained
- `monitor='val_loss'` — watches validation loss, not training loss
- `patience=10` — stops if val_loss doesn't improve for 10 consecutive epochs
- `restore_best_weights=True` — rolls back to the best checkpoint automatically
- **Result**: Training stopped at Epoch 17 — best val_loss was at Epoch 7

### TensorBoard
- Logs loss and accuracy curves per epoch
- `histogram_freq=1` logs weight distributions per epoch (useful for detecting dead neurons)
- View with: `tensorboard --logdir logs/fit`

### Training Results
```
Epoch 1:  val_loss=0.3459, val_accuracy=0.8650
Epoch 7:  val_loss=0.3384, val_accuracy=0.8645  ← best val_loss
Epoch 9:  val_loss=0.3491, val_accuracy=0.8685  ← best val_accuracy
Epoch 17: val_loss=0.3605, val_accuracy=0.8695
→ Early stopping triggered (patience=10 exceeded)
```

### Model Save
```python
model.save('model.h5')
```
Note: `.h5` is the legacy HDF5 format. Keras recommends `.keras` for new projects. This generates a deprecation warning during save but loads correctly.

---

## 8. Inference Pipeline (prediction.ipynb)

The inference notebook demonstrates the exact same pipeline the Streamlit app uses:

```python
# 1. Load saved artifacts
model = load_model('model.h5')
label_encoder_gender = pickle.load(open('label_encoder_gender.pkl', 'rb'))
label_encoder_geo    = pickle.load(open('one_hot_encoder_geo.pkl', 'rb'))
scaler               = pickle.load(open('scaler.pkl', 'rb'))

# 2. Create input as a DataFrame
input_data = {
    'CreditScore': 600, 'Geography': 'France', 'Gender': 'Male',
    'Age': 40, 'Tenure': 3, 'Balance': 60000,
    'NumOfProducts': 2, 'HasCrCard': 1,
    'IsActiveMember': 1, 'EstimatedSalary': 50000
}
input_df = pd.DataFrame([input_data])

# 3. Apply same preprocessing
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
geo_encoded = label_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_df = pd.DataFrame(geo_encoded,
    columns=label_encoder_geo.get_feature_names_out(['Geography']))
input_df = pd.concat([input_df.drop('Geography', axis=1), geo_df], axis=1)
input_scaled = scaler.transform(input_df)

# 4. Predict
prediction = model.predict(input_scaled)
# prediction = [[0.03225647]]  → 3.2% churn probability → NOT likely to churn
```

---

## 9. Streamlit Web App (app.py)

The app provides an interactive UI for real-time inference.

### Input Widgets
| Widget | Streamlit Component | Feature |
|---|---|---|
| Geography dropdown | `st.selectbox` (from encoder categories) | Geography |
| Gender dropdown | `st.selectbox` (from encoder classes) | Gender |
| Age | `st.slider(18, 92)` | Age |
| Balance | `st.number_input` | Balance |
| Credit Score | `st.number_input` | CreditScore |
| Estimated Salary | `st.number_input` | EstimatedSalary |
| Tenure | `st.slider(0, 10)` | Tenure |
| Num of Products | `st.slider(1, 4)` | NumOfProducts |
| Has Credit Card | `st.selectbox([0, 1])` | HasCrCard |
| Is Active Member | `st.selectbox([0, 1])` | IsActiveMember |

### Inference Flow in app.py
```python
# 1. Build input DataFrame from widget values
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    ...
})

# 2. OneHot encode Geography
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,
    columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# 3. Concatenate and scale
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

# 4. Predict and display
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]
st.write(f'Churn Probability: {prediction_probability:.2f}')
if prediction_probability > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is unlikely to churn.')
```

### Key Detail
The selectbox options are **dynamically loaded from the saved encoders**:
```python
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender    = st.selectbox('Gender',    label_encoder_gender.classes_)
```
This ensures the app always shows exactly the categories the model was trained on — no hardcoding.

---

## 10. Saved Artifacts Summary

| File | Created By | Used By | Contents |
|---|---|---|---|
| `model.h5` | `model.save()` | `app.py`, `prediction.ipynb` | Full trained ANN weights + architecture |
| `scaler.pkl` | `pickle.dump(scaler)` | `app.py`, `prediction.ipynb` | Fitted StandardScaler (mean/std per feature) |
| `label_encoder_gender.pkl` | `pickle.dump(label_encoder_gender)` | `app.py`, `prediction.ipynb` | Fitted LabelEncoder (Female→0, Male→1) |
| `one_hot_encoder_geo.pkl` | `pickle.dump(one_hot_encoder_geo)` | `app.py`, `prediction.ipynb` | Fitted OneHotEncoder (3 geography categories) |

---

## 11. Known Limitations

| Limitation | Details |
|---|---|
| No evaluation metrics printed | Test accuracy shown during training but no confusion matrix, AUC-ROC, or classification report was generated |
| No EDA | No class distribution plots, correlation heatmaps, or feature importance analysis |
| No hyperparameter tuning | Architecture (64→32), learning rate (0.01), and batch size are fixed without search |
| No regularization | No Dropout or L2 regularization layers — model may slightly overfit |
| Class imbalance unhandled | ~80/20 split not addressed with SMOTE or class_weight parameter |
| Legacy model format | `.h5` deprecated; Keras recommends `.keras` |

---

## 12. Common Interview Questions & Answers

**Q: Why did you use StandardScaler and not MinMaxScaler?**
A: StandardScaler (z-score normalization) is generally preferred for ANNs because it handles outliers better — MinMaxScaler compresses extreme values into [0,1] which can distort the distribution. StandardScaler centers the data at 0 with unit variance, which works well with gradient descent.

**Q: Why OneHotEncoder for Geography but LabelEncoder for Gender?**
A: LabelEncoder assigns integers (0, 1, 2...) which implies an ordinal relationship. For Gender (binary), this is fine — Female=0, Male=1 has no implied order that would mislead the model. For Geography (3 classes: France, Germany, Spain), using LabelEncoder would imply France < Germany < Spain, which is meaningless. OneHotEncoder creates independent binary columns for each category, removing false ordinal assumptions.

**Q: Why fit the scaler only on training data?**
A: Fitting on all data would cause data leakage — the scaler would compute mean/std from the test set, meaning the model indirectly "sees" test data statistics during training. This gives an overly optimistic evaluation. In production, the test set represents unseen future data, so you must scale it using statistics from training data only.

**Q: Why EarlyStopping with patience=10 instead of a fixed epoch count?**
A: A fixed epoch count risks underfitting (too few) or overfitting (too many). EarlyStopping monitors validation loss and stops when it stops improving, automatically finding the right number of epochs. `patience=10` gives the model 10 epochs to potentially recover from a temporary plateau before stopping. `restore_best_weights=True` ensures we use the checkpoint with lowest val_loss, not the final epoch.

**Q: What does the Sigmoid activation do in the output layer?**
A: Sigmoid maps any real number to (0, 1) via σ(x) = 1/(1+e^(-x)). This makes the output directly interpretable as a probability — the probability that the customer will churn. Values above 0.5 predict churn, below 0.5 predict staying.

**Q: What is Binary Crossentropy and why is it used?**
A: Binary Crossentropy measures the difference between the predicted probability and the actual label (0 or 1): Loss = -[y·log(ŷ) + (1-y)·log(1-ŷ)]. When the true label is 1 and the model predicts 0.99, loss is low. When it predicts 0.01, loss is very high. This heavily penalizes confident wrong predictions, driving the model to make well-calibrated probability estimates.
