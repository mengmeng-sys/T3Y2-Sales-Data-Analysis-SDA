# ===== generate_dataset.py =====
# PURPOSE: Create a fake (synthetic) sales dataset so we have data to work with.
# This is like making up practice data instead of using real company records.
# Run this file ONCE to create 'data/sales.csv' with 500 random sales.


# --- IMPORT STATEMENTS ---
# pandas  = lets us work with tables (DataFrames)
# numpy   = gives us random numbers and math tools
# datetime = handles dates (today, timedelta, etc.)
# random   = another way to pick random things
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# --- FIX THE RANDOM SEED ---
# A "seed" makes random numbers predictable.
# Without this, every run gives different data.
# With seed=42, we always get the SAME 500 records -- useful for debugging.
np.random.seed(42)
random.seed(42)

# --- PRODUCT CATALOG (Dictionary) ---
# A dictionary maps a KEY (product name) to a VALUE (category).
# Structure: { 'product_name': 'category_name' }
# We have 12 products across 3 categories.
products = {
    'Coke': 'Drink',
    'Pepsi': 'Drink',
    'Water': 'Drink',
    'Juice': 'Drink',
    'Bread': 'Food',
    'Rice': 'Food',
    'Pasta': 'Food',
    'Cereal': 'Food',
    'Soap': 'Household',
    'Detergent': 'Household',
    'Shampoo': 'Household',
    'Tissue': 'Household'
}

# --- GENERATE DATE RANGE ---
# We want sales data for the last 60 days.
# today()       = current date (e.g., 2026-07-08)
# timedelta(60) = a span of 60 days
# Subtracting 60 days from today gives us the start date.
today = datetime.now().date()
sixty_days_ago = today - timedelta(days=60)

# pd.date_range() creates a list of all dates between start and end.
# freq='D' means every day (daily frequency).
dates = pd.date_range(start=sixty_days_ago, end=today, freq='D')

# --- BUILD 500 SALES RECORDS ---
# We will generate 500 rows of data and store them in this list.
# Each row is itself a list: [Date, Product, Category, Quantity, UnitPrice, TotalSales]
all_sales = []

# Loop 500 times (underscore _ means we don't need the loop counter)
for _ in range(500):

    # Pick a random date from our list of 60 days
    sale_date = random.choice(dates)

    # Pick a random product name from the dictionary keys
    # list(products.keys()) gives ['Coke','Pepsi',...]
    product = random.choice(list(products.keys()))

    # Look up the category for that product
    # e.g., products['Coke'] returns 'Drink'
    category = products[product]

    # Random quantity: how many items the customer bought (1 to 10)
    qty = random.randint(1, 10)

    # --- DECIDE THE UNIT PRICE BASED ON CATEGORY ---
    # Different categories have different price ranges.
    # random.uniform(a, b) gives a random decimal between a and b.
    # round(..., 2) rounds to 2 decimal places (dollars and cents).
    if category == 'Drink':
        price = round(random.uniform(1.0, 3.0), 2)      # Drinks are cheap ($1-$3)
    elif category == 'Food':
        price = round(random.uniform(1.5, 5.0), 2)      # Food is mid-range
    else:  # Household
        price = round(random.uniform(2.0, 8.0), 2)      # Household items are most expensive

    # TotalSales = Quantity * UnitPrice
    # This is the amount of money for this one transaction
    total = round(qty * price, 2)

    # Add this row to our big list
    all_sales.append([sale_date, product, category, qty, price, total])

# --- CONVERT THE LIST INTO A DATAFRAME ---
# A DataFrame is like an Excel spreadsheet with rows and columns.
# We give it our list and name each column.
df = pd.DataFrame(
    all_sales,
    columns=['Date', 'Product', 'Category', 'Quantity', 'UnitPrice', 'TotalSales']
)

# --- SORT BY DATE ---
# sort_values('Date') arranges rows from oldest to newest.
# reset_index(drop=True) re-numbers the rows from 0 after sorting.
df = df.sort_values('Date').reset_index(drop=True)

# --- SAVE TO CSV ---
# CSV = Comma-Separated Values, a simple file format for tables.
# index=False means "don't write row numbers to the file".
df.to_csv('data/sales.csv', index=False)

# Print a confirmation message so user knows it worked
print(f"Created {len(df)} sales records in data/sales.csv")
print("\nFirst 5 rows:")
print(df.head())
