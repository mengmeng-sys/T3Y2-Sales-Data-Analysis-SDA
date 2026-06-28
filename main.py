# main.py
# This is our main program that orchestrates everything

import pandas as pd

# Import our cleaning functions
import data_cleaning as dc

print("="*50)
print("SALES DATA ANALYSIS PROJECT")
print("="*50)

# Step 1: Load and clean data
print("\nSTEP 1: DATA LOADING AND CLEANING")
print("-"*50)

# Load original data
df = dc.load_data()

# Clean the data
cleaned_df = dc.clean_data(df)

# Display cleaned data info
print("\n STEP 2: EXPLORING CLEANED DATA")
print("-"*50)

print("\nFirst 5 rows:")
print(cleaned_df.head())

print("\nDataset Info:")
print(cleaned_df.info())

print("\nStatistical Summary:")
print(cleaned_df.describe())

# Save cleaned data
dc.save_cleaned_data(cleaned_df)
