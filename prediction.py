# prediction.py
# This file handles feature engineering and machine learning

import pandas as pd
import numpy as np
from datetime import datetime

def load_clean_data():
    """Load the cleaned dataset"""
    df = pd.read_csv('data/sales_cleaned.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def create_features(df):
    """Create new features from existing data"""
    print("\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50)
    
    df_features = df.copy()
    
    # Date-based features
    df_features['DayOfWeek'] = df_features['Date'].dt.dayofweek
    df_features['Month'] = df_features['Date'].dt.month
    df_features['Day'] = df_features['Date'].dt.day
    df_features['IsWeekend'] = df_features['DayOfWeek'].isin([5, 6]).astype(int)
    df_features['DayName'] = df_features['Date'].dt.day_name()
    df_features['MonthName'] = df_features['Date'].dt.strftime('%B')
    df_features['Quarter'] = df_features['Date'].dt.quarter
    df_features['IsMonthStart'] = df_features['Day'].isin([1, 2, 3, 4, 5]).astype(int)
    df_features['IsMonthEnd'] = df_features['Day'].isin([25, 26, 27, 28, 29, 30, 31]).astype(int)
    df_features['RevenuePerItem'] = df_features['TotalSales'] / df_features['Quantity']
    
    print("✓ Created 10 new features")
    return df_features

def prepare_for_ml(df):
    """Prepare features for machine learning with proper encoding"""
    print("\n" + "="*50)
    print("PREPARING FOR MACHINE LEARNING")
    print("="*50)
    
    # One-Hot Encode categorical features
    print("\nEncoding categorical features...")
    
    # Encode Product
    product_dummies = pd.get_dummies(df['Product'], prefix='Product')
    print(f"   Product: Created {product_dummies.shape[1]} product columns")
    
    # Encode Category
    category_dummies = pd.get_dummies(df['Category'], prefix='Category')
    print(f"   Category: Created {category_dummies.shape[1]} category columns")
    
    # Numerical features
    numerical_features = ['Quantity', 'UnitPrice', 'DayOfWeek', 'Month', 'Day', 
                         'IsWeekend', 'Quarter', 'IsMonthStart', 'IsMonthEnd']
    
    # Combine everything
    X = pd.concat([
        df[numerical_features],
        product_dummies
    ], axis=1)
    
    y = df['TotalSales']
    
    print(f"\nFeatures (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Features: {X.columns.tolist()[:10]}... (and {X.shape[1]-10} more)")
    
    # Save prepared data
    prepared_data = pd.concat([X, y], axis=1)
    prepared_data.to_csv('data/prepared_data_with_products.csv', index=False)
    print("\n✓ Prepared data saved to 'data/prepared_data_with_products.csv'")
    
    return X, y

def run_feature_engineering():
    """Run all feature engineering steps"""
    print("="*50)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*50)
    
    df = load_clean_data()
    print(f"\nLoaded {len(df)} records")
    
    df_with_features = create_features(df)
    X, y = prepare_for_ml(df_with_features)
    
    print("\n" + "="*50)
    print("✓ FEATURE ENGINEERING COMPLETE")
    print("✓ Ready for machine learning!")
    print("="*50)
    
    return X, y, df_with_features

if __name__ == "__main__":
    run_feature_engineering()