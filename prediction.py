# ===== prediction.py =====
# PURPOSE: Build a Machine Learning model that predicts TotalSales.
#          This is the most advanced part of the project.
#
# PIPELINE: Load data -> Create features -> Split train/test -> Train model -> Evaluate -> Visualize
#
# WHAT IS MACHINE LEARNING HERE?
#   We give the computer lots of examples (sales records with features + the actual TotalSales).
#   The computer finds patterns and learns a formula (LinearRegression) to predict TotalSales
#   for new, unseen records.


import pandas as pd
import numpy as np
from datetime import datetime

# --- SKLEARN (Machine Learning Library) ---
# train_test_split  = splits data into training set and testing set
# LinearRegression  = the ML algorithm we'll use (finds a straight-line relationship)
# mean_absolute_error, mean_squared_error, r2_score = ways to measure how good our model is
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt  # for drawing charts


# ================================================
# PART 1: FEATURE ENGINEERING
# Feature engineering = creating new columns from existing data
# that might help the model make better predictions.
# ================================================


def load_clean_data():
    """Read the cleaned sales CSV."""
    df = pd.read_csv('data/sales_cleaned.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def create_features(df):
    """Add 10 new features (columns) derived from existing data.

    Why create features?
    The model needs numbers to learn from. A date like '2026-07-08' isn't a number,
    but features like DayOfWeek (0-6), Month (1-12), IsWeekend (0 or 1) ARE numbers
    that the model can understand.

    New features created:
    - DayOfWeek     (0=Monday, 6=Sunday)
    - Month         (1-12)
    - Day           (1-31)
    - IsWeekend     (1 if weekend, else 0)
    - DayName       ("Monday", "Tuesday", ...) — for display
    - MonthName     ("January", ...) — for display
    - Quarter       (1-4)
    - IsMonthStart  (1 if day is 1-5, else 0) — payday effect
    - IsMonthEnd    (1 if day is 25-31, else 0) — end-of-month effect
    - RevenuePerItem (TotalSales / Quantity) — avg price paid per item
    """
    print("\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50)

    f = df.copy()  # work on a copy to avoid changing original

    # .dt.dayofweek extracts the day number (0=Monday, 6=Sunday)
    f['DayOfWeek'] = f['Date'].dt.dayofweek

    # .dt.month gives month as a number (1=January, 12=December)
    f['Month'] = f['Date'].dt.month

    # .dt.day gives the day of the month (1-31)
    f['Day'] = f['Date'].dt.day

    # IsWeekend: 1 if DayOfWeek is 5 (Saturday) or 6 (Sunday), else 0
    # .isin([5,6]) checks if the value is in the list [5,6]
    # .astype(int) converts True/False to 1/0
    f['IsWeekend'] = f['DayOfWeek'].isin([5, 6]).astype(int)

    # For human-readable output (not used by the model, but nice for debugging)
    f['DayName'] = f['Date'].dt.day_name()
    f['MonthName'] = f['Date'].dt.strftime('%B')

    # Quarter of the year (1=Jan-Mar, 2=Apr-Jun, etc.)
    f['Quarter'] = f['Date'].dt.quarter

    # IsMonthStart: first 5 days of the month (people just got paid?)
    f['IsMonthStart'] = f['Day'].isin([1, 2, 3, 4, 5]).astype(int)

    # IsMonthEnd: last days of the month (people running out of money?)
    f['IsMonthEnd'] = f['Day'].isin([25, 26, 27, 28, 29, 30, 31]).astype(int)

    # RevenuePerItem = TotalSales / Quantity (how much each item costs on average)
    f['RevenuePerItem'] = f['TotalSales'] / f['Quantity']

    print("[OK] Created 10 new features")
    return f


def prepare_for_ml(df):
    """Build the feature matrix X and target vector y.

    X (features) = the INPUTS the model learns from
    y (target)   = the OUTPUT the model tries to predict

    For categorical data like 'Product' (Coke, Pepsi, etc.),
    we use one-hot encoding: create a separate column for EACH product,
    with value 1 if that product, 0 otherwise.
    """
    print("\n" + "="*50)
    print("PREPARING FOR MACHINE LEARNING")
    print("="*50)

    # --- ONE-HOT ENCODING ---
    # pd.get_dummies() converts 'Product' column into multiple binary columns.
    # Example: if Product='Coke', the Coke column = 1, all others = 0.
    # prefix='Product' makes column names like 'Product_Coke', 'Product_Pepsi', etc.
    product_dummies = pd.get_dummies(df['Product'], prefix='Product')
    print(f"   Product: Created {product_dummies.shape[1]} product columns")

    # Which numeric columns should the model use as inputs?
    numerical_features = [
        'Quantity', 'UnitPrice',      # original sales info
        'DayOfWeek', 'Month', 'Day',  # date-based features
        'IsWeekend', 'Quarter',       # more date features
        'IsMonthStart', 'IsMonthEnd'  # time-period features
    ]

    # Combine numeric features + one-hot encoded product columns
    # pd.concat() joins them side by side (axis=1 means columns)
    X = pd.concat([df[numerical_features], product_dummies], axis=1)

    # y = the column we want to PREDICT (TotalSales)
    y = df['TotalSales']

    print(f"\n[OK] Features (X) shape: {X.shape}  — {X.shape[1]} columns for {X.shape[0]} records")
    print(f"[OK] Target (y) shape: {y.shape}")

    return X, y


# ================================================
# PART 2: DATA SPLITTING
# We split our data into 2 sets:
#   - TRAINING set (70%): used to teach the model
#   - TESTING set (30%): used to check if the model learned well
# This way we know if the model works on data it hasn't seen before.
# ================================================


def split_data(X, y):
    """
    Split into train/test sets.

    test_size=0.3    -> 30% for testing, 70% for training
    random_state=42  -> fixed seed so we always get the same split
    """
    print("\n" + "="*50)
    print("SPLITTING DATA")
    print("="*50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"\nTraining (70%): X_train = {X_train.shape[0]} records, y_train = {y_train.shape[0]} records")
    print(f"Testing  (30%): X_test  = {X_test.shape[0]} records, y_test  = {y_test.shape[0]} records")

    return X_train, X_test, y_train, y_test


# ================================================
# PART 3: MODEL TRAINING
# LinearRegression tries to find the best STRAIGHT-LINE relationship
# between the features (X) and the target (y).
# Formula: y = w1*x1 + w2*x2 + ... + b
#   w1, w2... = weights (importance of each feature)
#   b         = bias (intercept)
# ================================================


def train_and_predict(X_train, X_test, y_train, y_test):
    """Train the LinearRegression model and make predictions on both sets."""
    print("\n" + "="*50)
    print("BUILDING MODEL")
    print("="*50)

    # Create the model (empty/unfitted)
    model = LinearRegression()

    # .fit() = TRAIN the model — it learns the best weights from training data
    model.fit(X_train, y_train)
    print("[OK] Model trained successfully!")

    # .predict() = use the trained model to guess TotalSales for new data
    y_pred_train = model.predict(X_train)  # predictions on training data
    y_pred_test  = model.predict(X_test)   # predictions on TEST data (unseen during training)
    print("[OK] Predictions made!")

    return model, y_pred_train, y_pred_test


# ================================================
# PART 4: MODEL EVALUATION
# We measure how good our predictions are using 4 metrics:
#
# MAE  (Mean Absolute Error):  Average error in dollars.
#                               If MAE = $3.50, predictions are off by ~$3.50 on average.
#
# MSE  (Mean Squared Error):   Like MAE but penalizes BIG errors more (squares them).
#
# RMSE (Root Mean Squared Error): Square root of MSE. In the same units as dollars.
#                                  More interpretable than MSE.
#
# R2   (R-squared / Coefficient of Determination):
#       Ranges from 0 to 1 (or negative if really bad).
#       0   = model explains nothing (as bad as guessing the average)
#       1   = perfect predictions
#       0.8 = model explains 80% of the variance in sales
# ================================================


def evaluate_model(y_test, y_pred_test):
    """Print evaluation metrics and save them to a text file."""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)

    # --- Calculate metrics ---
    mae  = mean_absolute_error(y_test, y_pred_test)
    mse  = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred_test)

    # --- Display results ---
    print(f"\nMAE  = ${mae:.2f}   -> Average prediction error in dollars")
    print(f"MSE  = ${mse:.2f}   -> Average squared error")
    print(f"RMSE = ${rmse:.2f}   -> Typical error (in dollars)")
    print(f"R2   =  {r2:.4f}     -> Model explains {r2*100:.1f}% of sales variance")

    # --- Interpretation ---
    print("\n" + "="*50)
    if r2 > 0.8:
        print("  [OK] Excellent model -- good for predictions")
    elif r2 > 0.6:
        print("  [OK] Good model -- use with some caution")
    elif r2 > 0.4:
        print("  [!] Decent model -- can be improved")
    else:
        print("  [!] Model needs improvement -- not reliable for predictions")

    # --- Save to file ---
    with open('data/model_metrics.txt', 'w') as f:
        f.write(f"MAE: ${mae:.2f}\nMSE: ${mse:.2f}\nRMSE: ${rmse:.2f}\nR2: {r2:.4f}\n")
    print("[OK] Metrics saved to 'data/model_metrics.txt'")

    return mae, mse, rmse, r2


# ================================================
# PART 5: VISUALIZING PREDICTIONS
# We draw a scatter plot:
#   x-axis = Actual TotalSales (the true value from the test set)
#   y-axis = Predicted TotalSales (what our model guessed)
#
# If the model is perfect, all points lie on the red dashed line (y = x).
# The closer the points are to that line, the better our model.
# ================================================


def visualize_predictions(y_test, y_pred_test):
    """Scatter plot: Actual vs Predicted values."""
    print("\n" + "="*50)
    print("VISUALIZING PREDICTIONS")
    print("="*50)

    plt.figure(figsize=(10, 6))

    # Scatter points: each point = one test record
    plt.scatter(y_test, y_pred_test, alpha=0.6, color='#2E86AB')
    # alpha=0.6 makes points slightly transparent (helps see overlapping points)

    # Perfect prediction line (red dashed)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect predictions')

    plt.xlabel('Actual TotalSales ($)')
    plt.ylabel('Predicted TotalSales ($)')
    plt.title('Actual vs Predicted Sales')
    plt.grid(True, alpha=0.3)

    # Add a text box with the metrics on the chart
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, f'R2 = {r2:.3f}\nMAE = ${mae:.2f}\nRMSE = ${rmse:.2f}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('data/predictions_scatter.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved visualization to 'data/predictions_scatter.png'")
    plt.show()


# ================================================
# PART 6: COMPLETE PIPELINE
# This function runs EVERYTHING in order.
# ================================================


def run_complete_pipeline():
    """Run the full ML pipeline from start to finish."""
    print("="*60)
    print("COMPLETE ML PIPELINE")
    print("(Load -> Features -> Split -> Train -> Evaluate -> Plot)")
    print("="*60)

    # Step 1: Load cleaned data
    df = load_clean_data()
    print(f"Loaded {len(df)} records")

    # Step 2: Create features
    df_with_features = create_features(df)

    # Step 3: Prepare X (features) and y (target)
    X, y = prepare_for_ml(df_with_features)

    # Step 4: Split into train/test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 5: Train model and get predictions
    model, y_pred_train, y_pred_test = train_and_predict(X_train, X_test, y_train, y_test)

    # Step 6: Evaluate the model
    mae, mse, rmse, r2 = evaluate_model(y_test, y_pred_test)

    # Step 7: Draw the scatter plot
    visualize_predictions(y_test, y_pred_test)

    # Print final summary
    print("\n" + "="*60)
    print(f"SUMMARY: MAE = ${mae:.2f}, RMSE = ${rmse:.2f}, R2 = {r2:.4f} ({r2*100:.1f}%)")
    print("[OK] COMPLETE! All files saved to 'data' folder")
    print("="*60)

    return model, X_train, X_test, y_train, y_test


# When run directly, execute the full pipeline
if __name__ == "__main__":
    run_complete_pipeline()
