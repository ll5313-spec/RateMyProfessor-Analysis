# Assessing Professor Effectiveness — RateMyProfessors Data Analysis

A data science capstone project for **Principles of Data Science II** (NYU, Spring 2025), analyzing a large-scale dataset scraped from [RateMyProfessors.com](https://www.ratemyprofessors.com/) to uncover patterns in how students evaluate professors.

---

## Project Overview

This project applies statistical analysis and machine learning to explore what factors drive professor ratings. The dataset contains **~90,000 professor records** with numerical and qualitative features including average rating, difficulty, gender, hotness ("pepper"), and teaching modality.

The analysis answers 10 research questions covering hypothesis testing, regression modeling, and classification, plus an extra credit exploration of regional variation.

---

## Research Questions & Methods

| # | Question | Method |
|---|----------|--------|
| Q1 | Is there a gender bias in professor ratings? | Mann-Whitney U test |
| Q2 | Does teaching experience affect rating quality? | Pearson correlation |
| Q3 | What is the relationship between difficulty and rating? | Pearson correlation |
| Q4 | Do online professors receive different ratings? | Mann-Whitney U test |
| Q5 | How does "Would Take Again" relate to ratings? | Pearson correlation |
| Q6 | Do "hot" professors receive higher ratings? | Mann-Whitney U test |
| Q7 | Can difficulty alone predict average rating? | Simple linear regression |
| Q8 | Can all factors together predict average rating? | Multiple linear regression + VIF |
| Q9 | Can rating predict "hot pepper" status? | Logistic regression + ROC/AUC |
| Q10 | Can all factors predict "hot pepper" status? | Logistic regression + ROC/AUC |
| EC | Are there regional patterns in hotness ratings? | State-level aggregation |

---

## Key Findings

- **Gender bias (Q1):** Male professors had a slightly higher median rating (4.2 vs 4.1); statistically significant (p = 0.00084), though effect size is small.
- **Difficulty & rating (Q3):** Strong negative correlation (r = −0.619); harder professors are rated lower.
- **"Would Take Again" (Q5):** Very strong positive correlation with ratings (r = 0.880).
- **Hotness & ratings (Q6):** "Hot" professors had a median rating of 4.5 vs 3.6 for others (p < 0.00001).
- **Multiple regression (Q8):** R² = 0.81, RMSE = 0.37, much stronger than difficulty-only model (R² = 0.38).
- **Pepper classification (Q10):** Full logistic model achieved AUC = 0.799 vs 0.778 for rating-only model.
- **Regional patterns (EC):** UK locations (Glasgow, Manchester, Surrey) showed disproportionately high hotness rates, suggesting cultural variation in how students use the platform.

---

## Tech Stack

- **Language:** Python 
- **Libraries:** `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `seaborn`

---

## Data

The dataset was provided by the course instructor (Professor Pascal Wallisch) and is not included in this repository due to course policy. The two source files used are:

- `rmpCapstoneNum.csv` — numerical features (rating, difficulty, gender, etc.)
- `rmpCapstoneQual.csv` — qualitative features (major, university, state)

**Preprocessing steps:**
- Manually assigned column names (no headers in raw files)
- Merged numerical and qualitative datasets
- Dropped rows with missing `Number of Ratings`
- Retained only professors with ≥ 5 ratings to reduce noise from extreme single-rating scores
- RNG seeded with student N-number for reproducibility

---

## Files

```
├── capstone_project_Helen_Li.py   # Full analysis code
├── Capstone_Project_Report.pdf    # Written report with findings and figures
└── README.md
```

---

## Course Info

- **Course:** Principles of Data Science II (DS UA 112), NYU
- **Instructor:** Professor Pascal Wallisch
- **Semester:** Spring 2025
