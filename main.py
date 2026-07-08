# ===== main.py =====
# This is the MAIN file - the one you run to start everything.
# Think of it like a manager: it tells each worker (other file) what to do in order.
# The 4 steps are: 1) Clean data, 2) Analyze numbers, 3) Draw charts, 4) Build ML model


# --- IMPORT LIBRARIES ---
# pandas = tool for working with tables (like Excel in Python)
import pandas as pd

# Import our OWN files (modules) so we can use their functions
import data_cleaning as dc       # handles loading & cleaning data
import analysis as analysis      # calculates statistics & insights
import visualization as viz      # draws charts
import prediction as pred        # handles machine learning model

# Print a fancy title so the user knows the program started
print("="*50)
print("SALES DATA ANALYSIS PROJECT")
print("="*50)

# =====================================================
# STEP 1: Load the raw data, clean it, and save result
# =====================================================
print("\nSTEP 1: DATA LOADING AND CLEANING")
print("-"*50)

# dc.load_data()    -> reads 'data/sales.csv' into a DataFrame (like a table)
df = dc.load_data()

# dc.clean_data(df) -> removes duplicates, handles missing values, fixes dates
# Returns a new cleaned DataFrame
cleaned_df = dc.clean_data(df)

# dc.save_cleaned_data(cleaned_df) -> writes the clean table to 'data/sales_cleaned.csv'
# We keep raw file unchanged and make a separate clean copy
dc.save_cleaned_data(cleaned_df)

# =====================================================
# STEP 2: Run business analysis on the cleaned data
# =====================================================
print("\nSTEP 2: ADVANCED DATA ANALYSIS")
print("-"*50)

# run_all_analysis() calls ALL the analysis functions inside analysis.py
# It prints: total revenue, best products, sales trends, insights, etc.
analysis.run_all_analysis()

# =====================================================
# STEP 3: Create visualizations (PNG chart files)
# =====================================================
print("\nSTEP 3: DATA VISUALIZATION")
print("-"*50)

# Reload the cleaned data again (each module loads its own copy)
viz_df = viz.load_data()

# Each function below creates a different chart and saves it as .png
viz.create_bar_chart(viz_df)    # horizontal bar: top products + category revenue
viz.create_line_chart(viz_df)   # line graph: daily sales over time
viz.create_pie_chart(viz_df)    # pie chart: revenue split by category
viz.create_histogram(viz_df)    # histograms: price & order-size distribution

# =====================================================
# STEP 4: Machine Learning Model
# =====================================================
print("\nSTEP 4: MACHINE LEARNING MODEL")
print("-"*50)

# run_complete_pipeline() does everything for ML:
# 1. Feature engineering (creates new features from dates)
# 2. Prepares data for ML (encodes products as numbers)
# 3. Splits data into training (70%) and testing (30%)
# 4. Trains Linear Regression model
# 5. Makes predictions
# 6. Evaluates the model (MAE, MSE, RMSE, R2)
# 7. Creates scatter plot comparing predictions vs actual
# 8. Saves metrics to 'model_metrics.txt'
# 9. Saves visualization to 'predictions_scatter.png'
model, X_train, X_test, y_train, y_test = pred.run_complete_pipeline()

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*50)
print("PROJECT COMPLETE!")
print("="*50)

print("\nAll files saved in the 'data' folder:")
print("   - sales.csv                    (Original raw data)")
print("   - sales_cleaned.csv            (Cleaned data)")
print("   - bar_charts.png               (Product & category chart)")
print("   - line_chart.png               (Daily sales trend)")
print("   - pie_chart.png                (Revenue distribution)")
print("   - histograms.png               (Price & quantity distribution)")
print("   - predictions_scatter.png      (ML predictions scatter plot)")
print("   - model_metrics.txt            (MAE, MSE, RMSE, R2 scores)")

print("\n" + "="*50)
print("END")
print("="*50)
