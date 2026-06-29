import pandas as pd
import numpy as np

def load_clean_data():
    """
    Load the cleaned dataset
    """
    df = pd.read_csv('data/sales_cleaned.csv')
    # Convert Date to datetime if not already
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def calculate_business_metrics(df):
    """
    Calculate important business metrics
    These tell us about the overall business performance
    """
    print("\n" + "="*50)
    print("📊 BUSINESS METRICS")
    print("="*50)
    
    # Total Revenue
    total_revenue = df['TotalSales'].sum()
    print(f"\n💰 Total Revenue: ${total_revenue:,.2f}")
    
    # Average Sales per Transaction
    avg_sales = df['TotalSales'].mean()
    print(f"📊 Average Sales per Transaction: ${avg_sales:.2f}")
    
    # Number of Transactions
    total_transactions = len(df)
    print(f"📝 Total Transactions: {total_transactions}")
    
    # Highest Single Sale
    highest_sale = df['TotalSales'].max()
    print(f"🏆 Highest Single Sale: ${highest_sale:.2f}")
    
    # Lowest Single Sale
    lowest_sale = df['TotalSales'].min()
    print(f"📉 Lowest Single Sale: ${lowest_sale:.2f}")
    
    # Total Quantity Sold
    total_quantity = df['Quantity'].sum()
    print(f"📦 Total Items Sold: {total_quantity}")
    
    # Average Quantity per Transaction
    avg_quantity = df['Quantity'].mean()
    print(f"📦 Average Items per Transaction: {avg_quantity:.2f}")

def find_best_products(df):
    """
    Find best selling products and categories
    """
    print("\n" + "="*50)
    print("🏆 BEST SELLING PRODUCTS & CATEGORIES")
    print("="*50)
    
    # Best Selling Product (by quantity)
    product_quantity = df.groupby('Product')['Quantity'].sum().sort_values(ascending=False)
    print("\n📦 Top 5 Products by Quantity Sold:")
    for i, (product, qty) in enumerate(product_quantity.head(5).items(), 1):
        print(f"   {i}. {product}: {qty} units")
    
    # Best Selling Product (by revenue)
    product_revenue = df.groupby('Product')['TotalSales'].sum().sort_values(ascending=False)
    print("\n💰 Top 5 Products by Revenue:")
    for i, (product, revenue) in enumerate(product_revenue.head(5).items(), 1):
        print(f"   {i}. {product}: ${revenue:,.2f}")
    
    # Best Selling Category (by quantity)
    category_quantity = df.groupby('Category')['Quantity'].sum().sort_values(ascending=False)
    print("\n📦 Top Categories by Quantity Sold:")
    for i, (category, qty) in enumerate(category_quantity.items(), 1):
        print(f"   {i}. {category}: {qty} units")
    
    # Best Selling Category (by revenue)
    category_revenue = df.groupby('Category')['TotalSales'].sum().sort_values(ascending=False)
    print("\n💰 Top Categories by Revenue:")
    for i, (category, revenue) in enumerate(category_revenue.items(), 1):
        print(f"   {i}. {category}: ${revenue:,.2f}")
    
    # Return these for later use
    return product_revenue, category_revenue

def analyze_sales_over_time(df):
    """
    Analyze sales patterns over time
    """
    print("\n" + "="*50)
    print("📈 SALES OVER TIME")
    print("="*50)
    
    # Daily Sales
    daily_sales = df.groupby('Date')['TotalSales'].sum()
    print(f"\n📅 Daily Sales Summary:")
    print(f"   Average Daily Sales: ${daily_sales.mean():.2f}")
    print(f"   Highest Daily Sales: ${daily_sales.max():.2f}")
    print(f"   Lowest Daily Sales: ${daily_sales.min():.2f}")
    
    # Monthly Sales
    # Extract month from date
    df['Month'] = df['Date'].dt.month
    monthly_sales = df.groupby('Month')['TotalSales'].sum()
    
    print(f"\n📅 Monthly Sales Summary:")
    for month, sales in monthly_sales.items():
        # Convert month number to name
        month_name = pd.to_datetime(f'2024-{month}-01').strftime('%B')
        print(f"   {month_name}: ${sales:,.2f}")
    
    # Weekly pattern (if we have enough data)
    # Extract day of week
    df['DayOfWeek'] = df['Date'].dt.day_name()
    daily_pattern = df.groupby('DayOfWeek')['TotalSales'].mean().sort_values(ascending=False)
    
    print(f"\n📅 Average Sales by Day of Week:")
    for day, avg in daily_pattern.items():
        print(f"   {day}: ${avg:.2f}")
    
    return daily_sales, monthly_sales

def find_insights(df):
    """
    Find interesting business insights
    """
    print("\n" + "="*50)
    print("💡 BUSINESS INSIGHTS")
    print("="*50)
    
    # Most common product
    most_common_product = df['Product'].mode()[0]
    print(f"\n1. Most Frequently Purchased Product: {most_common_product}")
    
    # Average price by category
    avg_price_by_category = df.groupby('Category')['UnitPrice'].mean()
    print("\n2. Average Price by Category:")
    for category, price in avg_price_by_category.items():
        print(f"   {category}: ${price:.2f}")
    
    # Revenue percentage by category
    category_revenue = df.groupby('Category')['TotalSales'].sum()
    total_revenue = df['TotalSales'].sum()
    revenue_percentage = (category_revenue / total_revenue) * 100
    
    print("\n3. Revenue Percentage by Category:")
    for category, percentage in revenue_percentage.items():
        print(f"   {category}: {percentage:.1f}%")
    
    # Price range analysis
    min_price = df['UnitPrice'].min()
    max_price = df['UnitPrice'].max()
    avg_price = df['UnitPrice'].mean()
    
    print(f"\n4. Price Range Analysis:")
    print(f"   Lowest Price: ${min_price:.2f}")
    print(f"   Highest Price: ${max_price:.2f}")
    print(f"   Average Price: ${avg_price:.2f}")
    
    # Quantity analysis
    min_qty = df['Quantity'].min()
    max_qty = df['Quantity'].max()
    avg_qty = df['Quantity'].mean()
    
    print(f"\n5. Quantity Analysis:")
    print(f"   Smallest Order: {min_qty} units")
    print(f"   Largest Order: {max_qty} units")
    print(f"   Average Order: {avg_qty:.1f} units")

def run_all_analysis():
    """
    Run all analysis functions
    This is the main function we'll call from main.py
    """
    print("="*50)
    print("📊 EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Load data
    df = load_clean_data()
    print(f"\nAnalyzing {len(df)} sales records...")
    
    # Run all analysis
    calculate_business_metrics(df)
    best_products = find_best_products(df)
    sales_over_time = analyze_sales_over_time(df)
    find_insights(df)
    
    print("\n" + "="*50)
    print("✅ Analysis Complete!")
    print("="*50)

# This runs when we execute this file directly
if __name__ == "__main__":
    run_all_analysis()