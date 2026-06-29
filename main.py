# main.py
# This is our main program that orchestrates everything

import pandas as pd

# Import our modules
import data_cleaning as dc
import analysis as analysis

print("="*50)
print("SALES DATA ANALYSIS PROJECT")
print("="*50)

# Step 1: Load and clean data
print("\n📊 STEP 1: DATA LOADING AND CLEANING")
print("-"*50)

# Load original data
df = dc.load_data()

# Clean the data
cleaned_df = dc.clean_data(df)

# Save cleaned data
dc.save_cleaned_data(cleaned_df)

# Step 2: Run Analysis
print("\n📊 STEP 2: EXPLORATORY DATA ANALYSIS")
print("-"*50)

# Load cleaned data for analysis
clean_df = analysis.load_clean_data()
analysis.calculate_business_metrics(clean_df)
analysis.find_best_products(clean_df)
analysis.analyze_sales_over_time(clean_df)
analysis.find_insights(clean_df)

print("\n" + "="*50)
print("✅ Project Analysis Complete!")
print("="*50)