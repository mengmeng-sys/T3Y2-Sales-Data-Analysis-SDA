# ===== data_cleaning.py =====
# PURPOSE: Load the raw CSV, inspect it, clean it, and save a clean copy.
# Cleaning = removing bad data so our analysis is accurate.
# Steps: drop duplicates, drop missing values, fix date format, reset index.


# --- IMPORTS ---
import pandas as pd  # for DataFrames and CSV reading/writing
import numpy as np   # for numerical operations (used in missing value checks)


# ================================================
# FUNCTION 1: load_data()
# WHAT IT DOES: Reads the raw CSV file into a DataFrame.
# RETURNS:      A pandas DataFrame containing all raw sales data.
# ================================================
def load_data():
    # pd.read_csv() opens a CSV file and converts it to a DataFrame automatically
    df = pd.read_csv('data/sales.csv')
    print("Data loaded successfully!")
    return df


# ================================================
# FUNCTION 2: display_initial_info(df)
# WHAT IT DOES: Prints basic info about the dataset BEFORE cleaning.
#               This helps us understand what needs fixing.
# PARAMETER df: The raw DataFrame.
# ================================================
def display_initial_info(df):
    print("\n"+"="*50)
    print("INITIAL DATA OVERVIEW")
    print("="*50)

    # .shape gives (row_count, column_count), like (500, 6)
    print(f"\n1. Data Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # .dtypes shows the data type of each column (object=text, int64=number, etc.)
    print("\n2. Data types:")
    print(df.dtypes)

    # .isnull() checks every cell: True if empty, False if filled.
    # .sum() adds up all the True values per column.
    print("\n3. Missing Values:")
    print(df.isnull().sum())

    # .duplicated() checks for duplicate rows (same data in every column).
    # .sum() counts how many total duplicate rows exist.
    print(f"\n4. Duplicate Rows: {df.duplicated().sum()}")


# ================================================
# FUNCTION 3: clean_data(df)
# WHAT IT DOES: Runs the full cleaning pipeline.
# PARAMETER df: Raw DataFrame.
# RETURNS:      A cleaned DataFrame.
#
# Cleaning steps:
#   1. Remove duplicate rows
#   2. Drop rows with any missing values
#   3. Convert Date column to proper date format
#   4. Reset row index (so numbers are sequential)
#   5. Print final info
# ================================================
def clean_data(df):
    print("\n" + "="*50)
    print("STARTING ADVANCED DATA CLEANING")
    print("="*50)

    # --- IMPORTANT: Make a COPY of the data ---
    # If we modify 'df' directly, pandas might give us a warning.
    # .copy() creates a completely separate copy so we can modify safely.
    df_clean = df.copy()

    # ---- STEP 1: Remove duplicate rows ----
    print("\n[1/5] Removing duplicates...")
    before_dup = len(df_clean)               # count rows BEFORE

    # drop_duplicates() removes rows that are exact copies of another row
    df_clean = df_clean.drop_duplicates()

    after_dup = len(df_clean)                # count rows AFTER
    print(f"   Removed {before_dup - after_dup} duplicate rows.")

    # ---- STEP 2: Handle missing values ----
    print("\n[2/5] Handling missing values...")

    # isnull() checks each cell, .sum() counts by column, .sum() again = total
    missing_count = df_clean.isnull().sum().sum()

    if missing_count > 0:
        print(f"   Found {missing_count} total missing values.")
        # dropna() removes ANY row that has at least one empty cell
        df_clean = df_clean.dropna()
        print("   Dropped rows with missing data.")
    else:
        print("   No missing values found.")

    # ---- STEP 3: Convert 'Date' column to datetime ----
    # The CSV stores dates as text (strings), but we need actual date objects
    # to do time-based analysis (group by month, find day of week, etc.)
    if 'Date' in df_clean.columns:
        print("\n[3/5] Converting 'Date' column to datetime...")

        # pd.to_datetime() converts text like '2026-01-15' into a date object.
        # errors='coerce' means: if a date is unreadable, set it to NaT (Not a Time)
        # instead of crashing the whole program.
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')

        # Check if any dates became NaT (unparseable)
        if df_clean['Date'].isnull().any():
            corrupted_count = df_clean['Date'].isnull().sum()
            # Drop the rows with bad dates (we can't fix them)
            df_clean = df_clean.dropna(subset=['Date'])
            print(f"   Dropped {corrupted_count} rows with bad dates.")
        print("   Date column standardised.")
    else:
        print("\n[3/5] Skip: 'Date' column not found.")

    # ---- STEP 4: Reset the index ----
    # After dropping rows, the row numbers (index) have gaps like 0,1,5,7...
    # reset_index(drop=True) re-numbers them 0,1,2,3... continuously.
    df_clean = df_clean.reset_index(drop=True)

    # ---- STEP 5: Print final verification ----
    print("\n[4/5] Final data types:")
    print("-" * 30)
    print(df_clean.dtypes)
    print("-" * 30)

    print("\n[5/5] Cleaning complete!")
    print(f"   Final shape: {df_clean.shape}")
    print("="*50 + "\n")

    return df_clean


# ================================================
# FUNCTION 4: save_cleaned_data(df)
# WHAT IT DOES: Writes the cleaned DataFrame to a new CSV file.
# PARAMETER df: The cleaned DataFrame.
# ================================================
def save_cleaned_data(df):
    # We save to a different filename ('sales_cleaned.csv') so the original stays safe
    save_to = 'data/sales_cleaned.csv'

    # to_csv() writes the DataFrame back to a CSV file
    # index=False means we don't include the row numbers in the file
    df.to_csv(save_to, index=False)
    print(f"\n Cleaned data saved to {save_to}")


# ================================================
# SPECIAL: if __name__ == "__main__"
# This block runs ONLY when we run THIS file directly.
# If this file is imported by main.py, this block is SKIPPED.
# ================================================
if __name__ == "__main__":
    print("="*50)
    print("DATA CLEANING SCRIPT")
    print("="*50)

    # Load the raw data from CSV
    df = load_data()

    # Show what the data looks like before cleaning
    display_initial_info(df)

    # Run the cleaning process
    cleaned_df = clean_data(df)

    # Show a preview of the cleaned data
    print("\n" + "="*50)
    print("AFTER CLEANING")
    print("="*50)
    print(cleaned_df.head())          # first 5 rows
    print("\nInfo after cleaning:")
    print(cleaned_df.info())          # summary of DataFrame

    # Save the cleaned version
    save_cleaned_data(cleaned_df)

    print("\n" + "="*50)
    print("Data cleaning complete!")
    print("="*50)
