# ===== analysis.py =====
# PURPOSE: Calculate business statistics and insights from the cleaned sales data.
#          This is where we answer questions like "What sells best?" and "When are sales highest?"
#
# ORGANIZATION:
#   - Basic functions (calculate_business_metrics, find_best_products, etc.)
#   - Advanced functions (monthly_comparison, trend_analysis, etc.)
#   - Orchestrator (run_all_analysis) that calls everything in order


import pandas as pd  # for DataFrames
import numpy as np   # for math operations


# ================================================
# HELPER FUNCTION: load_clean_data()
# Loads the cleaned CSV file so we can analyze it.
# Called by run_all_analysis() and also when running this file directly.
# ================================================
def load_clean_data():
    df = pd.read_csv('data/sales_cleaned.csv')
    # Convert the Date column from text to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])
    return df


# ================================================
# BASIC ANALYSIS FUNCTIONS
# ================================================


def calculate_business_metrics(df):
    """Print 7 key numbers that summarize overall business performance."""
    print("\n" + "="*50)
    print("BUSINESS METRICS")
    print("="*50)

    # .sum() adds up all values in a column
    total_revenue = df['TotalSales'].sum()
    print(f"\nTotal Revenue: ${total_revenue:,.2f}")  # :,.2f = format with commas & 2 decimals

    # .mean() calculates the average
    avg_sales = df['TotalSales'].mean()
    print(f"Average Sales per Transaction: ${avg_sales:.2f}")

    # len() counts how many rows in the DataFrame
    total_transactions = len(df)
    print(f"Total Transactions: {total_transactions}")

    # .max() finds the largest value
    highest_sale = df['TotalSales'].max()
    print(f"Highest Single Sale: ${highest_sale:.2f}")

    # .min() finds the smallest value
    lowest_sale = df['TotalSales'].min()
    print(f"Lowest Single Sale: ${lowest_sale:.2f}")

    # Total items sold across all transactions
    total_quantity = df['Quantity'].sum()
    print(f"Total Items Sold: {total_quantity}")

    # Average items per transaction
    avg_quantity = df['Quantity'].mean()
    print(f"Average Items per Transaction: {avg_quantity:.2f}")


def find_best_products(df):
    """Find which products and categories sell the most (by quantity and revenue)."""
    print("\n" + "="*50)
    print("BEST SELLING PRODUCTS AND CATEGORIES")
    print("="*50)

    # ---- Products by Quantity ----
    # groupby('Product') splits data into groups (one per product).
    # Then we sum the 'Quantity' column for each group.
    # sort_values(ascending=False) puts biggest first.
    product_quantity = df.groupby('Product')['Quantity'].sum().sort_values(ascending=False)
    print("\nTop 5 Products by Quantity Sold:")
    # .head(5) takes only the first 5 rows
    # .items() gives us (name, value) pairs
    # enumerate(..., 1) gives us 1,2,3... as the counter
    for i, (product, qty) in enumerate(product_quantity.head(5).items(), 1):
        print(f"   {i}. {product}: {qty} units")

    # ---- Products by Revenue (same logic, different column) ----
    product_revenue = df.groupby('Product')['TotalSales'].sum().sort_values(ascending=False)
    print("\nTop 5 Products by Revenue:")
    for i, (product, revenue) in enumerate(product_revenue.head(5).items(), 1):
        print(f"   {i}. {product}: ${revenue:,.2f}")

    # ---- Categories by Quantity ----
    category_quantity = df.groupby('Category')['Quantity'].sum().sort_values(ascending=False)
    print("\nTop Categories by Quantity Sold:")
    for i, (category, qty) in enumerate(category_quantity.items(), 1):
        print(f"   {i}. {category}: {qty} units")

    # ---- Categories by Revenue ----
    category_revenue = df.groupby('Category')['TotalSales'].sum().sort_values(ascending=False)
    print("\nTop Categories by Revenue:")
    for i, (category, revenue) in enumerate(category_revenue.items(), 1):
        print(f"   {i}. {category}: ${revenue:,.2f}")

    # Return the revenue data so other functions can use it too
    return product_revenue, category_revenue


def analyze_sales_over_time(df):
    """Examine daily, monthly, and day-of-week sales patterns."""
    print("\n" + "="*50)
    print("SALES OVER TIME")
    print("="*50)

    # ---- Daily Sales ----
    # Group by Date, sum TotalSales for each day
    daily_sales = df.groupby('Date')['TotalSales'].sum()
    print(f"\nDaily Sales Summary:")
    print(f"   Average Daily Sales: ${daily_sales.mean():.2f}")
    print(f"   Highest Daily Sales: ${daily_sales.max():.2f}")
    print(f"   Lowest Daily Sales: ${daily_sales.min():.2f}")

    # ---- Monthly Sales ----
    # Create a new column 'Month' by extracting the month number (1-12) from Date
    df['Month'] = df['Date'].dt.month
    monthly_sales = df.groupby('Month')['TotalSales'].sum()
    print(f"\nMonthly Sales Summary:")
    for month, sales in monthly_sales.items():
        # Convert month number (e.g., 7) to month name (e.g., "July")
        month_name = pd.to_datetime(f'2024-{month}-01').strftime('%B')
        print(f"   {month_name}: ${sales:,.2f}")

    # ---- Day-of-Week Pattern ----
    # .dt.day_name() gives "Monday", "Tuesday", etc.
    df['DayOfWeek'] = df['Date'].dt.day_name()
    # Group by day name, get average sales, sort highest first
    daily_pattern = df.groupby('DayOfWeek')['TotalSales'].mean().sort_values(ascending=False)
    print(f"\nAverage Sales by Day of Week:")
    for day, avg in daily_pattern.items():
        print(f"   {day}: ${avg:.2f}")

    return daily_sales, monthly_sales


def find_insights(df):
    """Extra interesting facts: most common product, prices by category, etc."""
    print("\n" + "="*50)
    print("BUSINESS INSIGHTS")
    print("="*50)

    # 1. Most frequently purchased product
    # .mode() returns the most common value(s), [0] gets the first one
    most_common_product = df['Product'].mode()[0]
    print(f"\n1. Most Frequently Purchased Product: {most_common_product}")

    # 2. Average price for each category
    avg_price_by_category = df.groupby('Category')['UnitPrice'].mean()
    print("\n2. Average Price by Category:")
    for category, price in avg_price_by_category.items():
        print(f"   {category}: ${price:.2f}")

    # 3. What percentage of total revenue does each category make?
    category_revenue = df.groupby('Category')['TotalSales'].sum()
    total_revenue = df['TotalSales'].sum()
    revenue_percentage = (category_revenue / total_revenue) * 100
    print("\n3. Revenue Percentage by Category:")
    for category, percentage in revenue_percentage.items():
        print(f"   {category}: {percentage:.1f}%")

    # 4. Price range (lowest, highest, average price)
    print(f"\n4. Price Range Analysis:")
    print(f"   Lowest Price: ${df['UnitPrice'].min():.2f}")
    print(f"   Highest Price: ${df['UnitPrice'].max():.2f}")
    print(f"   Average Price: ${df['UnitPrice'].mean():.2f}")

    # 5. Quantity range (smallest, largest, average order size)
    print(f"\n5. Quantity Analysis:")
    print(f"   Smallest Order: {df['Quantity'].min()} units")
    print(f"   Largest Order: {df['Quantity'].max()} units")
    print(f"   Average Order: {df['Quantity'].mean():.1f} units")


# ================================================
# ADVANCED ANALYSIS FUNCTIONS
# ================================================


def monthly_comparison(df):
    """Compare months side-by-side and calculate month-over-month growth rate."""
    print("\n" + "="*50)
    print("MONTHLY COMPARISON")
    print("="*50)

    # Extract month number and month name
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')

    # .agg() allows us to compute MULTIPLE aggregations at once:
    #   sum   = total revenue for the month
    #   mean  = average transaction value
    #   count = number of transactions
    # We also sum the Quantity column.
    monthly_metrics = df.groupby('Month').agg({
        'TotalSales': ['sum', 'mean', 'count'],
        'Quantity': 'sum'
    }).round(2)

    # The column names are multi-level after .agg(), so we flatten them
    monthly_metrics.columns = ['Total_Revenue', 'Avg_Transaction', 'Num_Transactions', 'Total_Items']

    print("\nMonthly Performance:")
    for month in monthly_metrics.index:
        m_name = pd.to_datetime(f'2024-{month}-01').strftime('%B')
        print(f"\n   {m_name}:")
        print(f"      Revenue: ${monthly_metrics.loc[month, 'Total_Revenue']:,.2f}")
        print(f"      Avg Transaction: ${monthly_metrics.loc[month, 'Avg_Transaction']:.2f}")
        print(f"      Transactions: {monthly_metrics.loc[month, 'Num_Transactions']}")
        print(f"      Items Sold: {monthly_metrics.loc[month, 'Total_Items']}")

    # ---- Month-over-Month Growth ----
    # Compares each month's revenue to the previous month
    if len(monthly_metrics) > 1:
        print("\nMonth-over-Month Growth:")
        for i in range(1, len(monthly_metrics)):
            cur_m = monthly_metrics.index[i]
            prev_m = monthly_metrics.index[i-1]
            # Growth formula: (current - previous) / previous * 100
            growth = ((monthly_metrics.loc[cur_m, 'Total_Revenue']
                      - monthly_metrics.loc[prev_m, 'Total_Revenue'])
                      / monthly_metrics.loc[prev_m, 'Total_Revenue']) * 100
            cur_name = pd.to_datetime(f'2024-{cur_m}-01').strftime('%B')
            prev_name = pd.to_datetime(f'2024-{prev_m}-01').strftime('%B')
            direction = "increase" if growth > 0 else "decrease"
            print(f"   {prev_name} -> {cur_name}: {growth:+.1f}% {direction}")

    return monthly_metrics


def product_performance_analysis(df):
    """Detailed breakdown of each product: total qty, revenue, avg price, revenue %."""
    print("\n" + "="*50)
    print("PRODUCT PERFORMANCE ANALYSIS")
    print("="*50)

    # group by Product, calculate multiple metrics at once
    pm = df.groupby('Product').agg({
        'Quantity': ['sum', 'mean'],      # total qty sold + avg qty per transaction
        'TotalSales': ['sum', 'mean'],    # total revenue + avg revenue per transaction
        'UnitPrice': 'mean'               # average unit price
    }).round(2)

    # Flatten multi-level column names into simple names
    pm.columns = ['Total_Quantity', 'Avg_Quantity', 'Total_Revenue', 'Avg_Revenue', 'Avg_Price']

    # Calculate each product's share of total revenue (as a percentage)
    total_rev = pm['Total_Revenue'].sum()
    pm['Revenue_Percentage'] = (pm['Total_Revenue'] / total_rev) * 100

    # Sort so the highest-revenue product appears first
    pm = pm.sort_values('Total_Revenue', ascending=False)

    print("\nAll Products Performance:")
    for product in pm.index:
        print(f"\n   {product}:")
        print(f"      Revenue: ${pm.loc[product, 'Total_Revenue']:,.2f} "
              f"({pm.loc[product, 'Revenue_Percentage']:.1f}%)")
        print(f"      Quantity: {pm.loc[product, 'Total_Quantity']:.0f} units")
        print(f"      Avg Price: ${pm.loc[product, 'Avg_Price']:.2f}")

    return pm


def top_10_products_analysis(df):
    """Focus on the top 10 products by revenue and their overall contribution."""
    print("\n" + "="*50)
    print("TOP 10 PRODUCTS ANALYSIS")
    print("="*50)

    pm = df.groupby('Product').agg({
        'Quantity': 'sum',
        'TotalSales': 'sum',
        'UnitPrice': 'mean'
    }).round(2)
    pm.columns = ['Total_Quantity', 'Total_Revenue', 'Avg_Price']

    # Sort by revenue descending and take the first 10
    top10 = pm.sort_values('Total_Revenue', ascending=False).head(10)

    total_rev = pm['Total_Revenue'].sum()
    total_qty = pm['Total_Quantity'].sum()

    print(f"\nTop 10 Products contribute:")
    print(f"   Revenue: ${top10['Total_Revenue'].sum():,.2f} "
          f"({top10['Total_Revenue'].sum()/total_rev*100:.1f}% of total)")
    print(f"   Quantity: {top10['Total_Quantity'].sum():.0f} units "
          f"({top10['Total_Quantity'].sum()/total_qty*100:.1f}% of total)")

    # Print each product with its details
    for i, p in enumerate(top10.index, 1):
        r = top10.loc[p, 'Total_Revenue']
        print(f"\n   {i}. {p}:")
        print(f"      Revenue: ${r:,.2f} ({r/total_rev*100:.1f}%)")
        print(f"      Quantity: {top10.loc[p, 'Total_Quantity']:.0f} units")
        print(f"      Avg Price: ${top10.loc[p, 'Avg_Price']:.2f}")

    return top10


def revenue_contribution_analysis(df):
    """Show how much each category and product contributes to total revenue (with visual bars)."""
    print("\n" + "="*50)
    print("REVENUE CONTRIBUTION ANALYSIS")
    print("="*50)

    total_revenue = df['TotalSales'].sum()

    # ---- Category Contribution ----
    cat_cont = df.groupby('Category')['TotalSales'].sum()
    cat_pct = (cat_cont / total_revenue) * 100
    print("\nRevenue Contribution by Category:")
    for cat in cat_cont.index:
        # Create a simple bar chart using = signs
        # Each '=' represents about 0.5% of revenue
        bar = '=' * int(cat_pct[cat] / 2)
        print(f"   {cat}: ${cat_cont[cat]:,.2f} ({cat_pct[cat]:.1f}%) {bar}")

    # ---- Product Contribution (top 5) ----
    prod_cont = df.groupby('Product')['TotalSales'].sum().sort_values(ascending=False)
    prod_pct = (prod_cont / total_revenue) * 100
    print("\nTop 5 Product Contributors:")
    for p in prod_cont.head(5).index:
        print(f"   {p}: ${prod_cont[p]:,.2f} ({prod_pct[p]:.1f}%)")

    # ---- Concentration Analysis ----
    # How much of the total revenue comes from just the top 5 or top 10 products?
    print(f"\nRevenue Concentration:")
    print(f"   Top 5 Products:  {prod_cont.head(5).sum()/total_revenue*100:.1f}% of total")
    print(f"   Top 10 Products: {prod_cont.head(10).sum()/total_revenue*100:.1f}% of total")

    return cat_cont, prod_cont


def trend_analysis(df):
    """Analyze weekly sales, find best/worst weeks, and check overall growth trend."""
    print("\n" + "="*50)
    print("TREND ANALYSIS")
    print("="*50)

    # Extract ISO week number from each date
    df['Week'] = df['Date'].dt.isocalendar().week
    weekly_sales = df.groupby('Week')['TotalSales'].sum()

    print("\nWeekly Sales Pattern:")
    for w in weekly_sales.index:
        print(f"   Week {w}: ${weekly_sales[w]:,.2f}")

    # Find the week with highest and lowest sales
    print(f"\nBest Week:  Week {weekly_sales.idxmax()} (${weekly_sales.max():,.2f})")
    print(f"Worst Week: Week {weekly_sales.idxmin()} (${weekly_sales.min():,.2f})")

    # Compare first week to last week to see overall trend
    if len(weekly_sales) > 2:
        change = ((weekly_sales.iloc[-1] - weekly_sales.iloc[0])
                  / weekly_sales.iloc[0]) * 100
        if change > 0:
            direction = "increased"
        elif change < 0:
            direction = "decreased"
        else:
            direction = "remained stable"
        print(f"\nOverall Trend: Sales {direction} by {abs(change):.1f}%")

    # Show how each category performed across months
    df['Month_Name'] = df['Date'].dt.strftime('%B')
    print("\nCategory Monthly Trends:")
    print(df.groupby(['Month_Name', 'Category'])['TotalSales'].sum().unstack().round(2))

    return weekly_sales


def create_analysis_summary(df):
    """Executive summary: key metrics, top performers, revenue split, recommendations."""
    print("\n" + "="*50)
    print("EXECUTIVE SUMMARY")
    print("="*50)

    total_revenue = df['TotalSales'].sum()
    total_transactions = len(df)
    avg_transaction = df['TotalSales'].mean()
    top_product_by_revenue = df.groupby('Product')['TotalSales'].sum().idxmax()
    top_product_by_quantity = df.groupby('Product')['Quantity'].sum().idxmax()
    top_category = df.groupby('Category')['TotalSales'].sum().idxmax()
    df['DayOfWeek'] = df['Date'].dt.day_name()
    busiest_day = df.groupby('DayOfWeek')['TotalSales'].sum().idxmax()

    print(f"\nKEY METRICS:")
    print(f"   Total Revenue: ${total_revenue:,.2f}")
    print(f"   Total Transactions: {total_transactions}")
    print(f"   Average Transaction: ${avg_transaction:.2f}")

    print("\nTOP PERFORMERS:")
    print(f"   Top Product (Revenue): {top_product_by_revenue}")
    print(f"   Top Product (Quantity): {top_product_by_quantity}")
    print(f"   Top Category: {top_category}")
    print(f"   Busiest Day: {busiest_day}")

    cat_rev = df.groupby('Category')['TotalSales'].sum()
    print("\nREVENUE DISTRIBUTION:")
    for cat in ['Drink', 'Food', 'Household']:
        print(f"   {cat}: {cat_rev.get(cat, 0)/total_revenue*100:.1f}%")

    print("\nKEY RECOMMENDATIONS (based on data):")
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


# ================================================
# ORCHESTRATOR: run_all_analysis()
# Calls every analysis function in sequence.
# ================================================
def run_all_analysis():
    print("="*50)
    print("ADVANCED DATA ANALYSIS")
    print("="*50)

    df = load_clean_data()
    print(f"\nAnalyzing {len(df)} sales records...")

    # ---- Basic ----
    calculate_business_metrics(df)
    find_best_products(df)
    analyze_sales_over_time(df)
    find_insights(df)

    # ---- Advanced ----
    monthly_comparison(df)
    product_performance_analysis(df)
    top_10_products_analysis(df)
    revenue_contribution_analysis(df)
    trend_analysis(df)

    # ---- Summary ----
    create_analysis_summary(df)

    print("\n" + "="*50)
    print("Advanced Analysis Complete")
    print("="*50)


# When this file is run directly (not imported), do the full analysis
if __name__ == "__main__":
    run_all_analysis()
