# ===== visualization.py =====
# PURPOSE: Create 4 types of charts from the cleaned sales data and save them as PNG images.
#          Charts help us SEE patterns that are hard to spot in raw numbers.
#
# CHARTS CREATED:
#   1. bar_charts.png   - Horizontal bar charts (top products + category revenue)
#   2. line_chart.png   - Daily sales trend with moving average
#   3. pie_chart.png    - Revenue distribution by category
#   4. histograms.png   - Distribution of prices and order quantities
#
# OUTPUT: All PNG files are saved to the 'data/' folder.


import pandas as pd               # for DataFrames
import matplotlib.pyplot as plt   # the main charting library
import numpy as np                # for math operations

# Set a visual style for all charts (dark grid background, nice colors)
plt.style.use('seaborn-v0_8-darkgrid')


def load_data():
    """Read the cleaned CSV file and convert Date to datetime."""
    df = pd.read_csv('data/sales_cleaned.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


# ================================================
# CHART 1: BAR CHARTS (2 side by side)
#
# Left chart:  Top 10 products by total revenue
# Right chart: Revenue by category (Drink, Food, Household)
#
# Horizontal bars are used because product names are text
# and easier to read when laid out horizontally.
# ================================================
def create_bar_chart(df):
    print("\nCreating Bar Charts...")

    # --- CREATE A FIGURE WITH 2 SUBPLOTS ---
    # subplots(1, 2) = 1 row, 2 columns of charts side by side
    # figsize=(14, 6) = width=14 inches, height=6 inches
    # ax1 = left chart, ax2 = right chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- LEFT CHART: Top 10 Products by Revenue ----
    # Group by Product, sum the TotalSales, sort ascending, take last 10 (highest)
    prod_rev = df.groupby('Product')['TotalSales'].sum().sort_values(ascending=True).tail(10)

    # barh() = horizontal bar chart (bars go left-to-right)
    # index = product names (y-axis), values = revenue amounts (x-axis)
    ax1.barh(prod_rev.index, prod_rev.values, color='skyblue')
    ax1.set_xlabel('Total Revenue ($)')
    ax1.set_title('Top 10 Products by Revenue', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)  # grid lines only on x-axis, semi-transparent

    # Add dollar labels at the end of each bar
    for i, v in enumerate(prod_rev.values):
        ax1.text(v + 5, i, f'${v:,.0f}', va='center', fontsize=9)

    # ---- RIGHT CHART: Revenue by Category ----
    cat_rev = df.groupby('Category')['TotalSales'].sum().sort_values(ascending=True)
    colors = ['#ff9999', '#66b3ff', '#99ff99']  # pink, blue, green
    ax2.barh(cat_rev.index, cat_rev.values, color=colors)
    ax2.set_xlabel('Total Revenue ($)')
    ax2.set_title('Revenue by Category', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    for i, v in enumerate(cat_rev.values):
        ax2.text(v + 5, i, f'${v:,.0f}', va='center', fontsize=10)

    # Adjust spacing so nothing overlaps, then save
    plt.tight_layout()
    plt.savefig('data/bar_charts.png', dpi=300, bbox_inches='tight')
    print("   Saved bar charts as 'data/bar_charts.png'")
    plt.show()


# ================================================
# CHART 2: LINE CHART (Daily Sales Trend)
#
# Shows how sales change from day to day over 60 days.
# Key features:
#   - Blue line with dots = actual daily sales
#   - Red dashed line = 7-day moving average (smooths out daily noise)
#   - Green arrow annotation = the day with highest sales
# ================================================
def create_line_chart(df):
    print("\nCreating Line Chart...")

    # Group sales by date and sum them up (one value per day)
    daily_sales = df.groupby('Date')['TotalSales'].sum()

    plt.figure(figsize=(12, 6))

    # Plot actual daily sales as a line with circle markers
    plt.plot(daily_sales.index, daily_sales.values,
             marker='o', linewidth=2, markersize=4, color='#2E86AB')

    # --- Moving Average (Trend Line) ---
    # rolling(window=7) takes a 7-day window and calculates the average.
    # This smooths out the zig-zags so we can see the overall trend.
    moving_avg = daily_sales.rolling(window=7).mean()
    plt.plot(daily_sales.index, moving_avg,
             linewidth=3, color='red', linestyle='--', label='7-Day Moving Average')

    # Chart labels and styling
    plt.title('Daily Sales Trend', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Sales ($)', fontsize=12)
    plt.legend()               # show the legend (identifies the red line)
    plt.grid(True, alpha=0.3)  # light grid
    plt.xticks(rotation=45)    # tilt date labels so they don't overlap

    # --- Annotation: Highest Sales Day ---
    # idxmax() finds the date with the maximum value
    max_date = daily_sales.idxmax()
    plt.annotate(f'Highest: ${daily_sales.max():.0f}',
                 xy=(max_date, daily_sales.max()),       # point to this location
                 xytext=(max_date, daily_sales.max() + 10),  # text placed slightly above
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=10)

    plt.tight_layout()
    plt.savefig('data/line_chart.png', dpi=300, bbox_inches='tight')
    print("   Saved line chart as 'data/line_chart.png'")
    plt.show()


# ================================================
# CHART 3: PIE CHART (Revenue Distribution by Category)
#
# Shows what percentage of total revenue comes from each category.
# Each slice = one category (Drink, Food, Household).
# The legend also shows the actual dollar amount.
# ================================================
def create_pie_chart(df):
    print("\nCreating Pie Chart...")

    cat_rev = df.groupby('Category')['TotalSales'].sum()

    plt.figure(figsize=(10, 8))

    # --- PIE CHART SETTINGS ---
    # autopct='%1.1f%%' shows percentage with 1 decimal (e.g., "33.3%")
    # explode=(0.05, 0.05, 0.05) = pull each slice slightly apart (for visual effect)
    # startangle=90 = first slice starts at the top
    # shadow=True = adds a drop shadow
    plt.pie(cat_rev.values,
            labels=cat_rev.index,
            colors=['#ff9999', '#66b3ff', '#99ff99'],  # pink, blue, green
            explode=(0.05, 0.05, 0.05),
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            textprops={'fontsize': 14})

    plt.title('Revenue Distribution by Category', fontsize=16, fontweight='bold')

    # Legend with category name + dollar amount
    plt.legend([f'{c}: ${v:,.0f}' for c, v in cat_rev.items()],
               loc='upper right', fontsize=10)

    # axis('equal') makes the pie a perfect circle (not an oval)
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig('data/pie_chart.png', dpi=300, bbox_inches='tight')
    print("   Saved pie chart as 'data/pie_chart.png'")
    plt.show()


# ================================================
# CHART 4: HISTOGRAMS (2 side by side)
#
# A histogram shows how data is DISTRIBUTED.
#   - x-axis = value buckets (bins)
#   - y-axis = how many times that value appears (frequency)
#
# Left:  Distribution of Unit Prices (most products are cheap?)
# Right: Distribution of Order Quantities (most people buy 1 item?)
#
# Vertical lines show the MEAN (red dashed) and MEDIAN (green dashed).
# If mean and median are close, the data is fairly balanced.
# ================================================
def create_histogram(df):
    print("\nCreating Histogram...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- LEFT HISTOGRAM: Price Distribution ----
    # bins=20 means 20 buckets spanning the price range
    ax1.hist(df['UnitPrice'], bins=20, color='#2E86AB', edgecolor='black', alpha=0.7)
    ax1.set_title('Distribution of Product Prices', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Unit Price ($)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # Add a vertical red dashed line for the MEAN (average) price
    ax1.axvline(df['UnitPrice'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: ${df["UnitPrice"].mean():.2f}')
    # Add a vertical green dashed line for the MEDIAN (middle) price
    ax1.axvline(df['UnitPrice'].median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: ${df["UnitPrice"].median():.2f}')
    ax1.legend()

    # ---- RIGHT HISTOGRAM: Quantity Distribution ----
    ax2.hist(df['Quantity'], bins=10, color='#A23B72', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribution of Order Sizes', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Quantity per Transaction')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)

    ax2.axvline(df['Quantity'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df["Quantity"].mean():.1f}')
    ax2.axvline(df['Quantity'].median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {df["Quantity"].median():.0f}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('data/histograms.png', dpi=300, bbox_inches='tight')
    print("   Saved histograms as 'data/histograms.png'")
    plt.show()


# ================================================
# ORCHESTRATOR: create_all_visualizations()
# Calls every chart function in order.
# ================================================
def create_all_visualizations():
    """Generate all 4 chart types and save them as PNG files."""
    print("="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)

    df = load_data()
    print(f"\nCreating charts from {len(df)} records...")

    create_bar_chart(df)
    create_line_chart(df)
    create_pie_chart(df)
    create_histogram(df)

    print("\n" + "="*50)
    print("All visualizations created!")
    print("Check the 'data' folder for your charts")
    print("="*50)


# When run directly, create all visualizations
if __name__ == "__main__":
    create_all_visualizations()
