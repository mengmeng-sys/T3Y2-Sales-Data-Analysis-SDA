# visualization.py
# This file creates all our charts and graphs
# We'll use matplotlib to make professional-looking visuals

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set a nice style for our charts
plt.style.use('seaborn-v0_8-darkgrid')

def load_data():
    """
    Load the cleaned dataset
    """
    df = pd.read_csv('data/sales_cleaned.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def create_bar_chart(df):
    """
    Create bar charts to compare product and category performance
    These help us see what's selling best
    """
    print("\n📊 Creating Bar Charts...")
    
    # Create figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Chart 1: Top 10 Products by Revenue
    product_revenue = df.groupby('Product')['TotalSales'].sum().sort_values(ascending=True).tail(10)
    
    # Create horizontal bar chart (easier to read with long names)
    ax1.barh(product_revenue.index, product_revenue.values, color='skyblue')
    ax1.set_xlabel('Total Revenue ($)')
    ax1.set_title('Top 10 Products by Revenue', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on the bars
    for i, v in enumerate(product_revenue.values):
        ax1.text(v + 5, i, f'${v:,.0f}', va='center', fontsize=9)
    
    # Chart 2: Revenue by Category
    category_revenue = df.groupby('Category')['TotalSales'].sum().sort_values(ascending=True)
    
    # Colors for different categories
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    bars = ax2.barh(category_revenue.index, category_revenue.values, color=colors)
    ax2.set_xlabel('Total Revenue ($)')
    ax2.set_title('Revenue by Category', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(category_revenue.values):
        ax2.text(v + 5, i, f'${v:,.0f}', va='center', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('data/bar_charts.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved bar charts as 'data/bar_charts.png'")
    plt.show()

def create_line_chart(df):
    """
    Create a line chart showing sales trends over time
    This helps us see patterns and seasonality
    """
    print("\n📈 Creating Line Chart...")
    
    # Calculate daily sales
    daily_sales = df.groupby('Date')['TotalSales'].sum()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot daily sales
    plt.plot(daily_sales.index, daily_sales.values, 
             marker='o', linewidth=2, markersize=4, color='#2E86AB')
    
    # Add a trend line (moving average)
    window = 7  # 7-day moving average
    moving_avg = daily_sales.rolling(window=window).mean()
    plt.plot(daily_sales.index, moving_avg, 
             linewidth=3, color='red', linestyle='--', 
             label=f'{window}-Day Moving Average')
    
    # Customize the chart
    plt.title('Daily Sales Trend', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Sales ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    
    # Add some annotations
    max_date = daily_sales.idxmax()
    max_value = daily_sales.max()
    plt.annotate(f'Highest: ${max_value:.0f}', 
                 xy=(max_date, max_value),
                 xytext=(max_date, max_value + 10),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=10)
    
    # Save the chart
    plt.tight_layout()
    plt.savefig('data/line_chart.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved line chart as 'data/line_chart.png'")
    plt.show()

def create_pie_chart(df):
    """
    Create a pie chart showing revenue distribution by category
    This helps us see the proportion of each category
    """
    print("\n🥧 Creating Pie Chart...")
    
    # Calculate revenue by category
    category_revenue = df.groupby('Category')['TotalSales'].sum()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Colors and explode (slightly separate slices)
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.05, 0.05, 0.05)  # Slightly separate all slices
    
    # Create pie chart
    plt.pie(category_revenue.values, 
            labels=category_revenue.index,
            colors=colors,
            explode=explode,
            autopct='%1.1f%%',  # Show percentages with 1 decimal
            startangle=90,
            shadow=True,
            textprops={'fontsize': 14})
    
    # Add title
    plt.title('Revenue Distribution by Category', fontsize=16, fontweight='bold')
    
    # Add a legend with actual values
    legend_labels = [f'{cat}: ${val:,.0f}' for cat, val in category_revenue.items()]
    plt.legend(legend_labels, loc='upper right', fontsize=10)
    
    # Make it a circle (not an oval)
    plt.axis('equal')
    
    # Save the chart
    plt.tight_layout()
    plt.savefig('data/pie_chart.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved pie chart as 'data/pie_chart.png'")
    plt.show()

def create_histogram(df):
    """
    Create a histogram showing the distribution of Unit Prices
    This helps us understand price ranges and patterns
    """
    print("\n📊 Creating Histogram...")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram 1: Price Distribution
    ax1.hist(df['UnitPrice'], bins=20, color='#2E86AB', edgecolor='black', alpha=0.7)
    ax1.set_title('Distribution of Product Prices', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Unit Price ($)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add vertical lines for mean and median
    mean_price = df['UnitPrice'].mean()
    median_price = df['UnitPrice'].median()
    ax1.axvline(mean_price, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_price:.2f}')
    ax1.axvline(median_price, color='green', linestyle='--', linewidth=2, label=f'Median: ${median_price:.2f}')
    ax1.legend()
    
    # Histogram 2: Quantity Distribution
    ax2.hist(df['Quantity'], bins=10, color='#A23B72', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribution of Order Sizes', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Quantity per Transaction', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add vertical lines for mean and median
    mean_qty = df['Quantity'].mean()
    median_qty = df['Quantity'].median()
    ax2.axvline(mean_qty, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_qty:.1f}')
    ax2.axvline(median_qty, color='green', linestyle='--', linewidth=2, label=f'Median: {median_qty:.0f}')
    ax2.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('data/histograms.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved histograms as 'data/histograms.png'")
    plt.show()

def create_all_visualizations():
    """
    Create all visualizations
    This is the main function we'll call
    """
    print("="*50)
    print("📊 CREATING VISUALIZATIONS")
    print("="*50)
    
    # Load data
    df = load_data()
    print(f"\nCreating charts from {len(df)} records...")
    
    # Create all charts
    create_bar_chart(df)
    create_line_chart(df)
    create_pie_chart(df)
    create_histogram(df)
    
    print("\n" + "="*50)
    print(" All visualizations created!")
    print(" Check the 'data' folder for your charts")
    print("="*50)

# This runs when we execute this file directly
if __name__ == "__main__":
    create_all_visualizations()