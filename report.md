# Sales Data Analysis and Prediction Using Python

## Project Report

**Student Name:** Seng Mengseang
**Course:** Automata Project  
**Date:** July 2025

---

## 1. Introduction

Data analysis has become essential for businesses to understand their customers and make better decisions. This project applies data science techniques to analyze sales data from a small shop and build a machine learning model to predict future sales.

The project follows the complete data science workflow:
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Data Visualization
5. Machine Learning
6. Model Evaluation

The dataset contains 500 sales records over 60 days. By analyzing this data, we can identify patterns, understand customer behavior, and make predictions that help the business grow.

---

## 2. Problem Statement

Small shop owners often struggle to understand their sales patterns. Without proper analysis, they face several challenges:

1. **Inventory Management:** Don't know which products to stock more of
2. **Staff Scheduling:** Don't know which days will be busiest
3. **Revenue Planning:** Can't predict future sales accurately
4. **Marketing Decisions:** Don't know which products to promote

This project addresses these problems by providing:
- Clear analysis of sales patterns
- Identification of best-selling products
- Understanding of daily and monthly trends
- A prediction model for future sales

---

## 3. Objectives

The main objectives of this project are:

1. **Collect and clean** sales data from a small shop
2. **Analyze** the data to find business insights
3. **Visualize** findings using charts and graphs
4. **Build a machine learning model** to predict future sales
5. **Evaluate** the model's performance
6. **Provide recommendations** to improve the business

---

## 4. Dataset Description

### 4.1 Data Source
The dataset was created using a Python script that generates realistic sales records. This approach was chosen because real sales data often contains sensitive information.

### 4.2 Data Structure
The dataset contains 500 sales records with 6 columns:

| Column | Description | Example |
|--------|-------------|---------|
| Date | Date of transaction | 2025-01-01 |
| Product | Name of product | Coke |
| Category | Product category | Drink |
| Quantity | Number of units | 5 |
| UnitPrice | Price per unit | $1.50 |
| TotalSales | Total amount (Quantity × UnitPrice) | $7.50 |

### 4.3 Products and Categories
The data includes 12 products across 3 categories:

**Drinks:** Coke, Pepsi, Water, Juice  
**Food:** Bread, Rice, Pasta, Cereal  
**Household:** Soap, Detergent, Shampoo, Tissue

### 4.4 Data Quality
- **Records:** 500
- **Time Period:** 60 days
- **No missing values:** All columns complete
- **No duplicates:** All rows unique
- **Data types:** Correct types for each column

---

## 5. Methodology

### 5.1 Data Cleaning

Before analysis, the data was cleaned to ensure accuracy:

| Step | Action | Result |
|------|--------|--------|
| 1 | Checked for duplicates | None found |
| 2 | Checked for missing values | None found |
| 3 | Converted Date to datetime | Correct date format |
| 4 | Verified data types | All correct |

**Code Used:**
```python
# Load the data
df = pd.read_csv('data/sales.csv')

# Remove duplicates
df = df.drop_duplicates()

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])
```
### 5.2 Exploratory Data Analysis (EDA)

The data was analyzed to find business insights:

**Business Metrics Calculated:**

- Total Revenue

- Average Sales per Transaction

- Best Selling Products (by quantity and revenue)

- Best Selling Categories

- Daily and Monthly Sales Patterns

- Day of Week Analysis

```python
total_revenue = df['TotalSales'].sum()
best_product = df.groupby('Product')['TotalSales'].sum().idxmax()
daily_sales = df.groupby('Date')['TotalSales'].sum()
```
### 5.3 Data Visualization

***Four types of charts were created:***

| Chart Type |	Purpose	| What It Shows |
| Bar Chart |	Compare categories |	Top products, category revenue |
| Line Chart |	Show trends |	Daily sales over time |
| Pie Chart |	Show proportions |	Revenue distribution |
| Histogram | Show distribution	| Price and quantity patterns |

**Code Used:**
```python
import matplotlib.pyplot as plt

# Bar chart
plt.barh(products, revenues)

# Line chart
plt.plot(dates, daily_sales)

# Pie chart
plt.pie(category_revenue, labels=categories, autopct='%1.1f%%')

# Histogram
plt.hist(df['UnitPrice'], bins=20)
```
### 5.4 Feature Engineering

New features were created to improve the machine learning model:

| Feature | Description | Reason |
|---------|-------------|--------|
| DayOfWeek | 0=Monday to 6=Sunday | Sales vary by day |
| Month | 1-12 | Sales vary by month |
| Day | 1-31 | Sales vary by date |
| IsWeekend | 1 if weekend, 0 if not | Weekend sales differ |
| Quarter | 1-4 | Seasonal patterns |
| IsMonthStart | First 5 days of month | Payday effect |
| IsMonthEnd | Last 7 days of month | Month-end effect |
| RevenuePerItem | TotalSales ÷ Quantity | Price per item |

### 5.5 Machine Learning Model

**Algorithm:** Linear Regression

**Why Linear Regression?**
- Simple and easy to understand
- Good for prediction problems
- Beginner-friendly

**Model Preparation:**

| Step | Description | Result |
|------|-------------|--------|
| 1 | One-Hot Encoding | Products converted to numbers |
| 2 | Train/Test Split | 70% training, 30% testing |
| 3 | Features | 21 features (9 numeric + 12 product) |
| 4 | Target | TotalSales |

**Code Used:**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## 6. Results

### 6.1 Analysis Results

**Business Metrics:**

| Metric | Value |
|--------|-------|
| Total Revenue | $9,908.96 |
| Average Transaction | $19.82 |
| Total Transactions | 500 |
| Total Items Sold | 2,781 |

**Best Selling Products:**

| Rank | By Quantity | Units | By Revenue | Revenue |
|------|-------------|-------|------------|---------|
| 1 | Coke | 255 | Detergent | $1,261.12 |
| 2 | Bread | 245 | Shampoo | $1,227.89 |
| 3 | Water | 243 | Rice | $930.45 |

**Best Selling Categories:**

| Category | Revenue | Percentage |
|----------|---------|------------|
| Household | $3,412.68 | 34.4% |
| Drinks | $3,248.14 | 32.8% |
| Food | $3,248.14 | 32.8% |

**Sales Patterns:**
- **Best Day:** Tuesday ($22.45 average)
- **Slowest Day:** Sunday ($16.98 average)
- **Sales Trend:** Increasing over time

### 6.2 Model Performance Results

**Evaluation Metrics:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean Absolute Error (MAE) | $3.94 | Predictions off by $3.94 on average |
| Mean Squared Error (MSE) | $31.35 | Average squared error |
| Root Mean Squared Error (RMSE) | $5.60 | Typical error is $5.60 |
| R² Score | 0.8735 | Model explains 87.4% of variance |

**Interpretation:**
-  R² = 87.4%: Excellent! Model explains most sales variation
-  MAE = $3.94: Good! Average error is small
-  RMSE = $5.60: Good! Typical error is reasonable

**Sample Predictions vs Actual:**

| Record | Actual | Predicted | Difference | Status |
|--------|--------|-----------|------------|--------|
| 1 | $19.36 | $18.92 | $0.44 |  Good |
| 2 | $11.26 | $10.87 | $0.39 |  Good |
| 3 | $8.55 | $8.21 | $0.34 |  Good |
| 4 | $78.98 | $45.67 | $33.31 |  Poor |

### 6.3 Key Insights

1. **80/20 Rule (Pareto Principle):**
   - Top 5 products generate 52.8% of revenue
   - Top 10 products generate 73.0% of revenue

2. **Product Types:**
   - Volume Products: Coke (sells many, low price)
   - Profit Products: Detergent (sells fewer, high price)

3. **Revenue Balance:**
   - Each category contributes about 33% of revenue
   - Business is well-balanced

4. **Growth Trend:**
   - Sales increased by 44.9% over 6 weeks
   - Business is healthy

5. **Busiest Day:**
   - Tuesday has highest sales
   - Need more staff on Tuesday

## 7. Conclusion

### 7.1 Summary

This project successfully analyzed sales data and built a prediction model for a small shop.

**Key Achievements:**
-  Analyzed 500 sales records
-  Identified best-selling products
-  Found sales patterns and trends
-  Built prediction model with 87.4% accuracy
-  Provided actionable business recommendations

### 7.2 Business Recommendations

Based on the analysis, the shop owner should:

1. **Focus on Detergent and Shampoo**
   - These generate the most revenue
   - Give them prime shelf space
   - Run promotions on them

2. **Keep Coke Well-Stocked**
   - It's the most popular product
   - Brings customers into the store
   - Even though it makes less money

3. **Schedule More Staff on Tuesdays**
   - This is the busiest day
   - Need to handle customer volume

4. **Maintain Balanced Inventory**
   - All categories perform well
   - Don't neglect any department

5. **Plan for Growth**
   - Sales are increasing
   - Order more inventory
   - Consider expanding

### 7.3 Model Usefulness

The model is good enough to use for:
-  Predicting daily sales
-  Planning inventory levels
-  Scheduling staff
-  Revenue forecasting

**Model Reliability:**
- 87.4% accuracy (R² score)
- $3.94 average error (MAE)
- $5.60 typical error (RMSE)

## 8. Future Improvements

### 8.1 Data Improvements

| Improvement | Description |
|-------------|-------------|
| More Data | Collect more records, longer time period |
| More Features | Add weather, holiday, promotion data |
| Customer Data | Add customer demographics |
| Product Details | Add more product attributes |

### 8.2 Model Improvements

| Improvement | Description |
|-------------|-------------|
| Different Algorithms | Random Forest, Decision Trees, XGBoost |
| Cross-Validation | Better model evaluation |
| Hyperparameter Tuning | Optimize model settings |
| Separate Models | One model per product/category |

### 8.3 Technical Improvements

| Improvement | Description |
|-------------|-------------|
| Web Interface | Web-based prediction system |
| Real-Time Dashboard | Live sales monitoring |
| Automated Reporting | Auto-generated reports |
| API | Integration with other systems |

### 8.4 Limitations

This project had some limitations:
- **Limited Data:** Only 500 records, 60 days
- **No External Factors:** Weather, holidays not included
- **Simple Model:** Only Linear Regression used
- **No Customer Data:** Customer behavior not tracked
- **Synthetic Data:** Not real customer data

## 9. References

- Python Documentation: https://docs.python.org/
- Pandas Documentation: https://pandas.pydata.org/
- Scikit-learn Documentation: https://scikit-learn.org/
- Matplotlib Documentation: https://matplotlib.org/
- Linear Regression Explanation: https://en.wikipedia.org/wiki/Linear_regression
- Course Materials: Introduction to Data Science, University Course

## 10. Appendix

### A. Complete File Structure

```
Sales_Project/
│
├── data/
│   ├── sales.csv
│   ├── sales_cleaned.csv
│   ├── prepared_data_with_products.csv
│   ├── bar_charts.png
│   ├── line_chart.png
│   ├── pie_chart.png
│   ├── histograms.png
│   ├── predictions_scatter.png
│   └── model_metrics.txt
│
├── main.py
├── data_cleaning.py
├── analysis.py
├── visualization.py
├── prediction.py
├── generate_dataset.py
├── requirements.txt
├── README.md
└── report.md
```

### B. Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Programming language |
| Pandas | 2.0.3 | Data manipulation |
| NumPy | 1.24.3 | Numerical operations |
| Matplotlib | 3.7.2 | Data visualization |
| Scikit-learn | 1.3.0 | Machine Learning |

### C. Code Snippets

**Data Cleaning:**

```python
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    return df
```

**Model Training:**

```python
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Model Evaluation:**

```python
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
```

