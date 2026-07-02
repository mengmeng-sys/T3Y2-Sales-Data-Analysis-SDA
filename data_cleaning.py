import pandas as pd
import numpy as np 

def load_data():
    """
    Load the sales dataset from CSV file
    Returns: DataFrame with the data
    """
    # Read the CSV file
    df = pd.read_csv('data/sales.csv')
    print("Data loaded successfully!")
    return df
def display_initial_info(df):
     """
    Show basic information about the dataset
    This helps us understand what needs cleaning
    """
     print("\n"+"="*50)
     print("INITIAL DATA OVERVIEW")
     # shape tell the row and the column of the data
     print(f"\n1. Data Shape: {df.shape[0]} rows, {df.shape[1]} columns")
     
     #data types
     
     print("\n2. Data types:")
     print(df.dtypes)
     
     #Missing values
     print("\n3. Missing Values:")
     print(df.isnull().sum())
     
     #Duplicates
     
     print(f"\n4. Duplicate Rows: {df.duplicated().sum()}")

def clean_data(df):
    """
    Main cleaning function.
    Performs robust, crash-proof cleaning operations.
    """
    print("\n" + "="*50)
    print("STARTING ADVANCED DATA CLEANING")
    print("="*50)
    
    # CRUCIAL FIX: Create an explicit copy to prevent SettingWithCopyWarning
    df_clean = df.copy()
    
    # Step 1: Remove duplicates
    print("\n[1/5] Removing duplicates...")
    before_dup = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after_dup = len(df_clean)
    print(f"   Success: Removed {before_dup - after_dup} duplicate rows.")
    
    # Step 2: Handle missing values
    print("\n[2/5] Handling missing values...")
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        print(f"   Warning: Found {missing_count} total missing values.")
        df_clean = df_clean.dropna()
        print("   Success: Dropped rows containing missing data.")
    else:
        print("   Success: No missing values found.")
        
    # Step 3: Convert Date to datetime safely
    if 'Date' in df_clean.columns:
        print("\n[3/5] Converting 'Date' column to datetime...")
        # FIX: errors='coerce' turns unparseable dates into NaT instead of crashing
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        
        # Drop any rows where the date was completely unparseable/corrupted
        if df_clean['Date'].isnull().any():
            corrupted_count = df_clean['Date'].isnull().sum()
            df_clean = df_clean.dropna(subset=['Date'])
            print(f"   Warning: Dropped {corrupted_count} rows with unparseable date formats.")
        print("   Success: 'Date' column successfully standardized.")
    else:
        print("\n[3/5] Skip: 'Date' column not found in this dataset.")
        
    # Step 4: Reset the index
    # FIX: Prevents broken/fragmented indexing after dropping rows
    df_clean = df_clean.reset_index(drop=True)
    
    # Step 5: Verify data types
    print("\n[4/5] Verifying final data types:")
    print("-" * 30)
    print(df_clean.dtypes)
    print("-" * 30)
    
    print("\n[5/5] Cleaning complete!")
    print(f"   Final dataset shape: {df_clean.shape}")
    print("="*50 + "\n")
    
    return df_clean
def save_cleaned_data(df):
    """
    save the cleaned dataset
    
    """
    # name the file that we want to save it 
    saveTo='data/sales_cleaned.csv'
    
    # save with new name to keep the original file safe
    df.to_csv(saveTo, index=False)
    print(f"\n Cleaned data saved to {saveTo}")
    
# This part runs when we execute this file directly
if __name__ == "__main__":
 
 print("="*50)
 print("DATA CLEANING SCRIPT")
 print("="*50)

 # load data first
 df = load_data()

 # Display initial info
 display_initial_info(df)

 # do data cleaning
 cleaned_df = clean_data(df)
 
 # display cleaned info 
 print("\n" + "="*50)
 print("AFTER CLEANING")
 print("="*50)
 print(cleaned_df.head())
 print("\nInfo after cleaning:")
 print(cleaned_df.info())
 
 # save cleaned data
 
 save_cleaned_data(cleaned_df)
 print("\n" + "="*50)
 print("✅ Data cleaning complete!")
 print("="*50)
    
    
    
    

    