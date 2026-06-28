# generate_dataset.py
# This creates our sales dataset

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for same results each time
np.random.seed(42)
random.seed(42)

# Our product catalog
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

# Generate dates for last 60 days
today = datetime.now().date()
sixty_days_ago = today - timedelta(days=60)
dates = pd.date_range(start=sixty_days_ago, end=today, freq='D')

# Create 500 sales records
all_sales = []

for _ in range(500):
    # Pick random date and product
    sale_date = random.choice(dates)
    product = random.choice(list(products.keys()))
    category = products[product]
    
    # Random quantity (1 to 10)
    qty = random.randint(1, 10)
    
    # Price based on category
    if category == 'Drink':
        price = round(random.uniform(1.0, 3.0), 2)
    elif category == 'Food':
        price = round(random.uniform(1.5, 5.0), 2)
    else:  # Household
        price = round(random.uniform(2.0, 8.0), 2)
    
    # Calculate total
    total = round(qty * price, 2)
    
    # Add to our list
    all_sales.append([sale_date, product, category, qty, price, total])

# Create DataFrame
df = pd.DataFrame(all_sales, columns=['Date', 'Product', 'Category', 
                                     'Quantity', 'UnitPrice', 'TotalSales'])

# Sort by date
df = df.sort_values('Date').reset_index(drop=True)

# Save to CSV
df.to_csv('data/sales.csv', index=False)

print(f"✅ Created {len(df)} sales records in data/sales.csv")
print("\nFirst 5 rows:")
print(df.head())