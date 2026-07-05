# analysis.py
# This file contains all our analysis functions
# I wrote this to help understand our sales data better

import pandas as pd
import numpy as np

def load_clean_data():
    """
    Load the cleaned dataset
    """
    df = pd.read_csv('data/sales_cleaned.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def calculate_business_metrics(df):
    """
    Calculate important business metrics
    These tell us about the overall business performance
    """
    print("\n" + "="*50)
    print("BUSINESS METRICS")
    print("="*50)
    
    # Total Revenue
    total_revenue = df['TotalSales'].sum()
    print(f"\nTotal Revenue: ${total_revenue:,.2f}")
    
    # Average Sales per Transaction
    avg_sales = df['TotalSales'].mean()
    print(f"Average Sales per Transaction: ${avg_sales:.2f}")
    
    # Number of Transactions
    total_transactions = len(df)
    print(f"Total Transactions: {total_transactions}")
    
    # Highest Single Sale
    highest_sale = df['TotalSales'].max()
    print(f"Highest Single Sale: ${highest_sale:.2f}")
    
    # Lowest Single Sale
    lowest_sale = df['TotalSales'].min()
    print(f"Lowest Single Sale: ${lowest_sale:.2f}")
    
    # Total Quantity Sold
    total_quantity = df['Quantity'].sum()
    print(f"Total Items Sold: {total_quantity}")
    
    # Average Quantity per Transaction
    avg_quantity = df['Quantity'].mean()
    print(f"Average Items per Transaction: {avg_quantity:.2f}")

def find_best_products(df):
    """
    Find best selling products and categories
    """
    print("\n" + "="*50)
    print("BEST SELLING PRODUCTS AND CATEGORIES")
    print("="*50)
    
    # Best Selling Product (by quantity)
    product_quantity = df.groupby('Product')['Quantity'].sum().sort_values(ascending=False)
    print("\nTop 5 Products by Quantity Sold:")
    for i, (product, qty) in enumerate(product_quantity.head(5).items(), 1):
        print(f"   {i}. {product}: {qty} units")
    
    # Best Selling Product (by revenue)
    product_revenue = df.groupby('Product')['TotalSales'].sum().sort_values(ascending=False)
    print("\nTop 5 Products by Revenue:")
    for i, (product, revenue) in enumerate(product_revenue.head(5).items(), 1):
        print(f"   {i}. {product}: ${revenue:,.2f}")
    
    # Best Selling Category (by quantity)
    category_quantity = df.groupby('Category')['Quantity'].sum().sort_values(ascending=False)
    print("\nTop Categories by Quantity Sold:")
    for i, (category, qty) in enumerate(category_quantity.items(), 1):
        print(f"   {i}. {category}: {qty} units")
    
    # Best Selling Category (by revenue)
    category_revenue = df.groupby('Category')['TotalSales'].sum().sort_values(ascending=False)
    print("\nTop Categories by Revenue:")
    for i, (category, revenue) in enumerate(category_revenue.items(), 1):
        print(f"   {i}. {category}: ${revenue:,.2f}")
    
    return product_revenue, category_revenue

def analyze_sales_over_time(df):
    """
    Analyze sales patterns over time
    """
    print("\n" + "="*50)
    print("SALES OVER TIME")
    print("="*50)
    
    # Daily Sales
    daily_sales = df.groupby('Date')['TotalSales'].sum()
    print(f"\nDaily Sales Summary:")
    print(f"   Average Daily Sales: ${daily_sales.mean():.2f}")
    print(f"   Highest Daily Sales: ${daily_sales.max():.2f}")
    print(f"   Lowest Daily Sales: ${daily_sales.min():.2f}")
    
    # Monthly Sales
    df['Month'] = df['Date'].dt.month
    monthly_sales = df.groupby('Month')['TotalSales'].sum()
    
    print(f"\nMonthly Sales Summary:")
    for month, sales in monthly_sales.items():
        month_name = pd.to_datetime(f'2024-{month}-01').strftime('%B')
        print(f"   {month_name}: ${sales:,.2f}")
    
    # Weekly pattern
    df['DayOfWeek'] = df['Date'].dt.day_name()
    daily_pattern = df.groupby('DayOfWeek')['TotalSales'].mean().sort_values(ascending=False)
    
    print(f"\nAverage Sales by Day of Week:")
    for day, avg in daily_pattern.items():
        print(f"   {day}: ${avg:.2f}")
    
    return daily_sales, monthly_sales

def find_insights(df):
    """
    Find interesting business insights
    """
    print("\n" + "="*50)
    print("BUSINESS INSIGHTS")
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

# ===== ADVANCED ANALYSIS FUNCTIONS =====

def monthly_comparison(df):
    """
    Compare monthly sales performance
    This helps identify seasonal trends and growth
    """
    print("\n" + "="*50)
    print("MONTHLY COMPARISON")
    print("="*50)
    
    # Extract month and year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')
    
    # Calculate monthly metrics
    monthly_metrics = df.groupby('Month').agg({
        'TotalSales': ['sum', 'mean', 'count'],
        'Quantity': 'sum'
    }).round(2)
    
    # Flatten column names
    monthly_metrics.columns = ['Total_Revenue', 'Avg_Transaction', 'Num_Transactions', 'Total_Items']
    
    print("\nMonthly Performance:")
    for month in monthly_metrics.index:
        month_name = pd.to_datetime(f'2024-{month}-01').strftime('%B')
        revenue = monthly_metrics.loc[month, 'Total_Revenue']
        avg = monthly_metrics.loc[month, 'Avg_Transaction']
        transactions = monthly_metrics.loc[month, 'Num_Transactions']
        items = monthly_metrics.loc[month, 'Total_Items']
        
        print(f"\n   {month_name}:")
        print(f"      Revenue: ${revenue:,.2f}")
        print(f"      Average Transaction: ${avg:.2f}")
        print(f"      Number of Transactions: {transactions}")
        print(f"      Total Items Sold: {items}")
    
    # Calculate month-over-month growth
    if len(monthly_metrics) > 1:
        print("\nMonth-over-Month Growth:")
        for i in range(1, len(monthly_metrics)):
            current_month = monthly_metrics.index[i]
            prev_month = monthly_metrics.index[i-1]
            
            current_rev = monthly_metrics.loc[current_month, 'Total_Revenue']
            prev_rev = monthly_metrics.loc[prev_month, 'Total_Revenue']
            
            growth = ((current_rev - prev_rev) / prev_rev) * 100
            
            current_name = pd.to_datetime(f'2024-{current_month}-01').strftime('%B')
            prev_name = pd.to_datetime(f'2024-{prev_month}-01').strftime('%B')
            
            if growth > 0:
                print(f"   {prev_name} to {current_name}: +{growth:.1f}% increase")
            else:
                print(f"   {prev_name} to {current_name}: {growth:.1f}% decrease")
    
    return monthly_metrics

def product_performance_analysis(df):
    """
    Detailed product performance analysis
    """
    print("\n" + "="*50)
    print("PRODUCT PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Calculate product metrics
    product_metrics = df.groupby('Product').agg({
        'Quantity': ['sum', 'mean'],
        'TotalSales': ['sum', 'mean'],
        'UnitPrice': 'mean'
    }).round(2)
    
    # Rename columns
    product_metrics.columns = ['Total_Quantity', 'Avg_Quantity', 'Total_Revenue', 'Avg_Revenue', 'Avg_Price']
    
    # Calculate revenue percentage
    total_revenue = product_metrics['Total_Revenue'].sum()
    product_metrics['Revenue_Percentage'] = (product_metrics['Total_Revenue'] / total_revenue) * 100
    
    # Sort by revenue
    product_metrics = product_metrics.sort_values('Total_Revenue', ascending=False)
    
    print("\nAll Products Performance:")
    for product in product_metrics.index:
        revenue = product_metrics.loc[product, 'Total_Revenue']
        quantity = product_metrics.loc[product, 'Total_Quantity']
        avg_price = product_metrics.loc[product, 'Avg_Price']
        percentage = product_metrics.loc[product, 'Revenue_Percentage']
        
        print(f"\n   {product}:")
        print(f"      Total Revenue: ${revenue:,.2f} ({percentage:.1f}% of total)")
        print(f"      Total Quantity: {quantity:.0f} units")
        print(f"      Average Price: ${avg_price:.2f}")
    
    return product_metrics

def top_10_products_analysis(df):
    """
    Analyze top 10 products in detail
    """
    print("\n" + "="*50)
    print("TOP 10 PRODUCTS ANALYSIS")
    print("="*50)
    
    # Calculate metrics
    product_metrics = df.groupby('Product').agg({
        'Quantity': ['sum'],
        'TotalSales': ['sum'],
        'UnitPrice': 'mean'
    }).round(2)
    
    product_metrics.columns = ['Total_Quantity', 'Total_Revenue', 'Avg_Price']
    
    # Get top 10 by revenue
    top_10 = product_metrics.sort_values('Total_Revenue', ascending=False).head(10)
    
    # Calculate percentage of total
    total_revenue = product_metrics['Total_Revenue'].sum()
    total_quantity = product_metrics['Total_Quantity'].sum()
    
    top_10_revenue = top_10['Total_Revenue'].sum()
    top_10_quantity = top_10['Total_Quantity'].sum()
    
    print(f"\nTop 10 Products contribute:")
    print(f"   Revenue: ${top_10_revenue:,.2f} ({top_10_revenue/total_revenue*100:.1f}% of total)")
    print(f"   Quantity: {top_10_quantity:.0f} units ({top_10_quantity/total_quantity*100:.1f}% of total)")
    
    print("\nDetailed Top 10:")
    for i, product in enumerate(top_10.index, 1):
        revenue = top_10.loc[product, 'Total_Revenue']
        quantity = top_10.loc[product, 'Total_Quantity']
        price = top_10.loc[product, 'Avg_Price']
        rev_pct = (revenue / total_revenue) * 100
        
        print(f"\n   {i}. {product}:")
        print(f"      Revenue: ${revenue:,.2f} ({rev_pct:.1f}% of total)")
        print(f"      Quantity: {quantity:.0f} units")
        print(f"      Average Price: ${price:.2f}")
    
    return top_10

def revenue_contribution_analysis(df):
    """
    Analyze revenue contribution by product and category
    """
    print("\n" + "="*50)
    print("REVENUE CONTRIBUTION ANALYSIS")
    print("="*50)
    
    total_revenue = df['TotalSales'].sum()
    
    # Category contribution
    category_contribution = df.groupby('Category')['TotalSales'].sum()
    category_percentage = (category_contribution / total_revenue) * 100
    
    print("\nRevenue Contribution by Category:")
    for category in category_contribution.index:
        revenue = category_contribution[category]
        percentage = category_percentage[category]
        # Simple visual bar using = signs
        bar = '=' * int(percentage / 2)
        print(f"   {category}: ${revenue:,.2f} ({percentage:.1f}%) {bar}")
    
    # Product contribution (top contributors)
    product_contribution = df.groupby('Product')['TotalSales'].sum().sort_values(ascending=False)
    product_percentage = (product_contribution / total_revenue) * 100
    
    print("\nTop 5 Product Contributors:")
    for product in product_contribution.head(5).index:
        revenue = product_contribution[product]
        percentage = product_percentage[product]
        print(f"   {product}: ${revenue:,.2f} ({percentage:.1f}%)")
    
    # Concentration analysis
    top_5_products_revenue = product_contribution.head(5).sum()
    top_10_products_revenue = product_contribution.head(10).sum()
    
    print(f"\nRevenue Concentration:")
    print(f"   Top 5 Products: {top_5_products_revenue/total_revenue*100:.1f}% of total revenue")
    print(f"   Top 10 Products: {top_10_products_revenue/total_revenue*100:.1f}% of total revenue")
    
    return category_contribution, product_contribution

def trend_analysis(df):
    """
    Identify trends and patterns in the data
    """
    print("\n" + "="*50)
    print("TREND ANALYSIS")
    print("="*50)
    
    # Weekly trends
    df['Week'] = df['Date'].dt.isocalendar().week
    weekly_sales = df.groupby('Week')['TotalSales'].sum()
    
    print("\nWeekly Sales Pattern:")
    for week in weekly_sales.index:
        sales = weekly_sales[week]
        print(f"   Week {week}: ${sales:,.2f}")
    
    # Identify best and worst weeks
    best_week = weekly_sales.idxmax()
    worst_week = weekly_sales.idxmin()
    
    print(f"\nBest Week: Week {best_week} (${weekly_sales[best_week]:,.2f})")
    print(f"Worst Week: Week {worst_week} (${weekly_sales[worst_week]:,.2f})")
    
    # Growth trend
    if len(weekly_sales) > 2:
        first_week_sales = weekly_sales.iloc[0]
        last_week_sales = weekly_sales.iloc[-1]
        overall_change = ((last_week_sales - first_week_sales) / first_week_sales) * 100
        
        print(f"\nOverall Trend (Week 1 to Week {len(weekly_sales)}):")
        if overall_change > 0:
            print(f"   Sales increased by {overall_change:.1f}%")
        elif overall_change < 0:
            print(f"   Sales decreased by {abs(overall_change):.1f}%")
        else:
            print("   Sales remained stable")
    
    # Category trends over time
    df['Month_Name'] = df['Date'].dt.strftime('%B')
    category_monthly = df.groupby(['Month_Name', 'Category'])['TotalSales'].sum().unstack()
    
    print("\nCategory Monthly Trends:")
    print(category_monthly.round(2))
    
    return weekly_sales

def create_analysis_summary(df):
    """
    Create a comprehensive summary of all analysis
    This is like an executive summary
    """
    print("\n" + "="*50)
    print("EXECUTIVE SUMMARY")
    print("="*50)
    
    # Key metrics
    total_revenue = df['TotalSales'].sum()
    total_transactions = len(df)
    avg_transaction = df['TotalSales'].mean()
    
    # Best performers
    top_product_by_revenue = df.groupby('Product')['TotalSales'].sum().idxmax()
    top_product_by_quantity = df.groupby('Product')['Quantity'].sum().idxmax()
    top_category = df.groupby('Category')['TotalSales'].sum().idxmax()
    
    # Busiest day
    df['DayOfWeek'] = df['Date'].dt.day_name()
    busiest_day = df.groupby('DayOfWeek')['TotalSales'].sum().idxmax()
    
    print("\nKEY METRICS:")
    print(f"   Total Revenue: ${total_revenue:,.2f}")
    print(f"   Total Transactions: {total_transactions}")
    print(f"   Average Transaction: ${avg_transaction:.2f}")
    
    print("\nTOP PERFORMERS:")
    print(f"   Top Product (Revenue): {top_product_by_revenue}")
    print(f"   Top Product (Quantity): {top_product_by_quantity}")
    print(f"   Top Category: {top_category}")
    print(f"   Busiest Day: {busiest_day}")
    
    # Revenue distribution
    category_revenue = df.groupby('Category')['TotalSales'].sum()
    drink_pct = (category_revenue.get('Drink', 0) / total_revenue) * 100
    food_pct = (category_revenue.get('Food', 0) / total_revenue) * 100
    household_pct = (category_revenue.get('Household', 0) / total_revenue) * 100
    
    print("\nREVENUE DISTRIBUTION:")
    print(f"   Drinks: {drink_pct:.1f}%")
    print(f"   Food: {food_pct:.1f}%")
    print(f"   Household: {household_pct:.1f}%")
    
    # Recommendations
    print("\nKEY RECOMMENDATIONS:")
    print("   1. Focus marketing on Detergent and Shampoo (highest revenue)")
    print("   2. Stock up on Coke (highest volume)")
    print("   3. Consider Tuesday promotions (busiest day)")
    print("   4. Maintain balanced inventory across all categories")
    
    return {
        'total_revenue': total_revenue,
        'total_transactions': total_transactions,
        'avg_transaction': avg_transaction,
        'top_product': top_product_by_revenue,
        'top_category': top_category
    }

def run_all_analysis():
    """
    Run all analysis functions including advanced ones
    """
    print("="*50)
    print("ADVANCED DATA ANALYSIS")
    print("="*50)
    
    # Load data
    df = load_clean_data()
    print(f"\nAnalyzing {len(df)} sales records...")
    
    # Basic analysis (from Day 3)
    calculate_business_metrics(df)
    find_best_products(df)
    analyze_sales_over_time(df)
    find_insights(df)
    
    # Advanced analysis
    monthly_comparison(df)
    product_performance_analysis(df)
    top_10_products_analysis(df)
    revenue_contribution_analysis(df)
    trend_analysis(df)
    
    # Summary
    create_analysis_summary(df)
    
    print("\n" + "="*50)
    print("Advanced Analysis Complete")
    print("="*50)

# This runs when we execute this file directly
if __name__ == "__main__":
    run_all_analysis()