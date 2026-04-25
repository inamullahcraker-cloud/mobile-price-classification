# 📱 Mobile Price Classification (ML Pipeline Project)

This project builds a **machine learning pipeline** to classify mobile phones into different **price ranges (0–3)** based on their specifications.

---

## 🚀 Project Overview

The goal is to predict mobile price categories using features like:

* RAM
* Battery power
* Camera specs (fc, pc)
* Screen size & resolution
* Connectivity (4G, WiFi, etc.)

The project focuses on **end-to-end pipeline design**, including preprocessing, transformation, training, and model saving.

---

## 🧠 Key Highlights

* 📊 Exploratory Data Analysis (EDA) with histograms & boxplots
* ⚙️ Feature preprocessing using **ColumnTransformer & Pipeline**
* 🔄 Handling skewed features using custom transformation (√x)
* 🧪 Stratified train-test split
* 🤖 Model training with proper evaluation

---

## 🏗️ Pipeline Design

### 🔹 Numerical Features

* Missing values → `SimpleImputer (median)`
* Scaling → `StandardScaler`

### 🔹 Skewed Feature Handling

* Applied **square root transformation**:

```python
def skewness(x):
    return np.sqrt(x)
```

* Then scaled for normalization

---

## 📊 Model Evaluation

* Classification Report (Precision, Recall, F1-score)
* Confusion Matrix (Train & Test)
* Achieved approx:

  * ✅ **~93–96% accuracy**

---

## 💾 Model Saving

Two approaches used:


* Models stored with date:

```
models/mobile_model_YYYY-MM-DD.pkl
```

Includes:

* Pipeline
* Accuracy
* Feature metadata

---

## 📂 Project Structure

```bash
├── Moble_price_detecting.ipynb   # Main notebook
├── models/                       # Saved models (versioned)
├── train.csv                     # Dataset
├── README.md                    # Documentation
                
```

---

## ▶️ How to Run

```bash
git clone https://github.com/inamullahcraker-cloud/mobile-price-classifiaction.git
cd your-repo-name
pip install -r requirements.txt
jupyter notebook
```

---

## 📌 Future Improvements

* Add hyperparameter tuning (GridSearchCV fully structured)
* Try advanced models (XGBoost, LightGBM)
* Deploy as API or web app
* Add feature importance visualization

---

## 👨‍💻 Author

**Inamullah**

---

## 📄 License

Open-source (MIT)
