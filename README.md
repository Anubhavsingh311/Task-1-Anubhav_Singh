# Task-1-Anubhav_Singh
### DecodeLabs Data Science Internship | Project 1 — Advanced EDA & Feature Engineering

**Goal:** Transform raw, chaotic e-commerce orders data into a mathematically clean dataset ready for machine learning algorithms.

---

## Repository Structure

```
Task-1-Anubhav_Singh/
│
├── Project1.ipynb            ← main notebook
├── dataset.csv               ← raw dataset (1200 rows, 14 columns)
├── cleaned_dataset.csv       ← final cleaned + feature-engineered output
├── README.md
└── images/
    ├── missing_values.png
    ├── distributions.png
    ├── engineered_features.png
    └── correlation_heatmap.png
```

---

## Key Requirements Covered

- **Missing Data** — statistical imputation via KNN (Mean/Median/KNN)
- **Outlier Detection & Treatment** — IQR (Interquartile Range)
- **Feature Engineering** — 3+ new predictive features from existing columns
- **Key Skills** — Pandas, NumPy, statistical analysis, data cleaning, feature extraction

---

## Dataset

E-commerce orders dataset with 1200 rows and 14 columns: `OrderID`, `Date`, `CustomerID`, `Product`, `Quantity`, `UnitPrice`, `TotalPrice`, `PaymentMethod`, `OrderStatus`, `CouponCode`, `ReferralSource`, and more.

Only `CouponCode` had missing values — 25.75% of rows.

---

## What I did

### 1. Exploratory Data Analysis
- Checked data types, shape, and basic statistics
- Identified missing values and their percentage per column
- Plotted distributions across all numeric columns

### 2. Missing Value Handling — KNN Imputation
`CouponCode` had 25.75% missing values. Decision logic:
- < 5% missing → drop rows
- 5–20% missing → median / group imputation
- **> 20% missing → KNN Imputation** ← applied here

Encoded categorical columns numerically, ran `KNNImputer(n_neighbors=5)`, then decoded back to original category names.

### 3. Outlier Detection & Treatment
Used the **IQR method** to find outliers across all numeric columns:
- Lower Bound = Q1 − 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR

Applied **Winsorization** (`np.clip`) to cap extreme values at the boundary — no rows were dropped.

### 4. Feature Engineering

| Feature | Logic | Why |
|---|---|---|
| `CartUtilizationRate` | Quantity / ItemsInCart | Purchase conversion per cart |
| `HasCoupon` | 1 if coupon used, else 0 | Coupon impact on order value |
| `OrderDayOfWeek` | day of week from Date | Weekly purchase patterns |
| `IsWeekend` | 1 if Sat/Sun, else 0 | Weekend vs weekday behaviour |
| `OrderMonth` | month from Date | Seasonal trends |
| `RevenuePerCartItem` | TotalPrice / ItemsInCart | Revenue efficiency per cart slot |

### 5. Correlation Analysis
Heatmap revealed `UnitPrice` and `Quantity` as the strongest predictors of `TotalPrice`. `CartUtilizationRate` showed moderate correlation with `Quantity`.

---

## Tools Used

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (KNNImputer, LabelEncoder)

---

*DecodeLabs Data Science Internship | Batch 2026*
