# 🏃‍♂️ ML Running Performance Scorer

**Analyze your running form using Machine Learning trained on ~177,000 real-world activities.**

This project uses **Principal Component Analysis (PCA)** to evaluate a runner's performance based on a `.FIT` file (standard format from Garmin, Suunto, Coros, etc.). Unlike simple calculators, this tool compares your metrics against a massive dataset to determine your "Performance Score" on a scale of **0-10**.

---

## 🧠 How it works

The model analyzes the relationship between your **Speed** and your **Physiological Cost (Heart Rate)**, while considering biomechanics (Cadence, Stride Length) and terrain.

### The Dataset
The AI "Brain" (`performance_scorer.pkl`) was trained on **177,696 unique running activities**. This ensures the scoring system is statistically robust and covers a wide range of runner levels, from beginners to elites.

### The "Smart" Logic
The algorithm evaluates runs based on weighted features (determined by PCA weights):

* **Pace (min/km):** The strongest indicator. Faster is better (negative correlation in PCA).
* **Efficiency Index (Speed / HR):** High speed at low heart rate yields the highest points.
* **Stride Length:** Longer effective stride correlates with better performance.
* **Cadence:** Rewards optimal turnover (typically ~170-180 spm).
* **Context:** Adjusts for Elevation Gain and Distance.

---

## 🚀 Quick Start

### 1. Prerequisites
You need **Python 3.8+** and the following libraries. You can install them using pip:

```bash
pip install pandas numpy scikit-learn fitparse joblib seaborn matplotlib
```

### 2. Score your run
You can use the provided sample files (`*.fit`) or your own file exported from Garmin Connect/Strava.

**Option A: Interactive Mode (Drag & Drop)**
Simply run the script and drag your file into the terminal when prompted:

```bash
python score_my_run.py
```

**Option B: Command Line Argument**
Run the script passing the filename directly:

```bash
python score_my_run.py your_run_file.fit
```

---

## 📊 Interpreting the Score

The score is a relative ranking against the population of ~177k runs. It is not just a measure of speed, but of **efficiency**.

| Score | Level | Description |
| :--- | :--- | :--- |
| **8.0 - 10.0** | **Elite / Master** | Exceptional pace with very low cardiac drift. Top percentile of the dataset. |
| **6.0 - 7.9** | **Advanced** | Strong aerobic base and good mechanics. Well above average. |
| **4.0 - 5.9** | **Solid Amateur** | Typical recreational runner. Balanced performance. |
| **0.0 - 3.9** | **Beginner / Recovery** | Developing base, recovery run, or heavy terrain difficulty. |

---

## 📂 Project Structure

* `score_my_run.py` - Main script. Loads the model, parses your .FIT file, and calculates the score.
* `performance_scorer.pkl` - The pre-trained AI model (Scaler + PCA weights).
* `clean_data.py` - Script used to clean and filter raw CSV data (removes GPS errors, ultra-short runs, walking, etc.).
* `train_model.py` - Script used to train the PCA algorithm on the cleaned dataset.
* `*.fit` - Sample running files included for testing purposes.

> **Note:** The raw training dataset (`*.csv`) is excluded from this repository due to file size and privacy reasons.

---

## 🛠 Tech Stack

* **Python** (Data Processing)
* **Pandas** (Data Manipulation)
* **Scikit-Learn** (PCA, StandardScaler, MinMaxScaler)
* **Fitparse** (Binary .FIT file parsing)


_Created by Jan Zubalewicz & Mateusz Kubita_
