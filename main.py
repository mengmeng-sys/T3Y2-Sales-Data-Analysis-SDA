# main.py
# This is our main program that orchestrates everything

import pandas as pd

# Import our modules
import data_cleaning as dc
import analysis as analysis
import visualization as viz

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

# Step 3: Create Visualizations
print("\n📊 STEP 3: DATA VISUALIZATION")
print("-"*50)

# Load data for visualization
viz_df = viz.load_data()
viz.create_bar_chart(viz_df)
viz.create_line_chart(viz_df)
viz.create_pie_chart(viz_df)
viz.create_histogram(viz_df)

print("\n" + "="*50)
print("✅ Project Complete!")
print("📁 All charts saved in the 'data' folder")
print("="*50)