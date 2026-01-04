# 🏃‍♂️ ML Running Performance Scorer

> **Analyze your running form using Machine Learning trained on ~177,000 real-world activities.**

This project uses **Principal Component Analysis (PCA)** to evaluate a runner's performance based on a `.FIT` file (standard format from Garmin, Suunto, Coros, etc.). Unlike simple calculators, this tool compares your metrics against a massive dataset to determine your **"Performance Score"** on a scale of **0-10**.

---

## 🧠 How it works

The model does not rely on simple formulas. Instead, it uses **Unsupervised Learning (PCA)** to find patterns in a dataset of over **170,000 runs** sourced from the *GoldenCheetah OpenData* project.

It evaluates the relationship between your **Output** (Pace) and your **Cost** (Heart Rate), adjusted for your personal context (Age, Weight, Gender) and biomechanics.

### The "Raw Physics" Logic
The algorithm evaluates runs based on **10 weighted features**:

* **Pace (min/km):** The primary performance metric. Faster is better.
* **Heart Rate (bpm):** The physiological cost. Lower HR at a given pace = higher score.
* **Aerobic Decoupling:** Measures cardiac drift (fatigue). Stability over time is rewarded.
* **Cadence:** Rewards optimal turnover (typically ~170-180 spm).
* **Stride Length:** Indicates power and efficiency.
* **Context:** The score is adjusted for Age, Weight, Gender, Elevation Gain, and Distance.

---

## 🚀 Quick Start

### 1. Prerequisites
You need **Python 3.8+** and the following libraries:

```bash
pip install pandas numpy scikit-learn fitparse joblib seaborn matplotlib goldencheetah-opendata
```

### 2. (Optional) Train your own model
If you want to rebuild the AI model from scratch:

1.  **Fetch & Clean Data:**
    ```bash
    python clean_data.py
    ```
    *This will automatically fetch data using `fetch_data.py` and clean it.*

2.  **Train Model:**
    ```bash
    python train_model.py
    ```
    *This trains the PCA and saves `performance_scorer.pkl`.*

### 3. Score your run
To evaluate your run, simply use the scoring script. You can drag & drop your `.FIT` file into the terminal when prompted.

```bash
python score_my_run.py
```

---

## 📊 Interpreting the Score

The score is a relative ranking against the population of runners in the dataset.

| Score | Level | Description |
| :--- | :--- | :--- |
| **8.0 - 10.0** | **Elite / Master** | Exceptional pace relative to heart rate and age. Top percentile. |
| **6.0 - 7.9** | **Advanced** | Strong aerobic base and mechanics. Well above average. |
| **4.0 - 5.9** | **Solid Amateur** | Typical recreational runner. Balanced performance. |
| **0.0 - 3.9** | **Beginner / Recovery** | Developing base, recovery run, or heavy terrain difficulty. |

---

## 📂 Project Structure

* `score_my_run.py` - **Main Application.** Loads the model, parses your `.FIT` file, asks for user context (Age/Weight), and calculates the score.
* `performance_scorer.pkl` - The pre-trained AI model (Scaler + PCA weights).
* `fetch_data.py` - Downloads raw activity metadata using the `goldencheetah-opendata` library.
* `clean_data.py` - Cleans the raw data, calculates derived metrics (Decoupling, Stride), and prepares `ready_to_train.csv`.
* `train_model.py` - Trains the PCA algorithm, determines feature weights, and saves the model.
* `*.fit` - Sample running files included for testing purposes.

> **Note:** The raw training dataset (`activities.csv`) is excluded from the repository. It will be downloaded automatically if you run the training scripts.

---

## 🛠 Tech Stack

* **Python** (Data Processing)
* **Pandas** (Data Manipulation)
* **Scikit-Learn** (PCA, StandardScaler, MinMaxScaler)
* **Fitparse** (Binary .FIT file parsing)
* **GoldenCheetah OpenData** (Data Source)

---

Created by **Jan Zubalewicz** and **Mateusz Kubita**
