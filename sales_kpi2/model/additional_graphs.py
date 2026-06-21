import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_additional_graphs(df_full):
    """
    Generate additional analytical graphs for better insights.
    df_full should have columns: customer_name, product, amount, date, region
    """
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#4f46e5', '#7c3aed', '#a855f7', '#10b981', '#f59e0b', '#ef4444']
    
    # 1. Monthly Sales Trend
    df_full['date'] = pd.to_datetime(df_full['date'])
    df_full['month'] = df_full['date'].dt.to_period('M')
    monthly_sales = df_full.groupby('month')['amount'].sum().reset_index()
    monthly_sales['month'] = monthly_sales['month'].astype(str)
    
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_sales['month'], monthly_sales['amount'], 
             marker='o', linewidth=3, markersize=10, color='#10b981')
    plt.fill_between(range(len(monthly_sales)), monthly_sales['amount'], 
                     alpha=0.3, color='#10b981')
    plt.title('Monthly Sales Trend', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Sales (₹)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'monthly_trend.png'), dpi=100)
    plt.close()
    
    # 2. Top 5 Customers by Revenue
    top_customers = df_full.groupby('customer_name')['amount'].sum().nlargest(5).reset_index()
    
    plt.figure(figsize=(10, 5))
    bars = plt.barh(top_customers['customer_name'], top_customers['amount'], 
                    color=colors[:5])
    plt.title('Top 5 Customers by Revenue', fontsize=16, fontweight='bold')
    plt.xlabel('Total Revenue (₹)', fontsize=12)
    plt.ylabel('Customer', fontsize=12)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'₹{width:,.0f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'top_customers.png'), dpi=100)
    plt.close()
    
    # 3. Product Performance Comparison
    product_stats = df_full.groupby('product').agg({
        'amount': ['sum', 'count', 'mean']
    }).reset_index()
    product_stats.columns = ['product', 'total_sales', 'quantity_sold', 'avg_price']
    product_stats = product_stats.sort_values('total_sales', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total sales by product
    ax1.bar(product_stats['product'], product_stats['total_sales'], color=colors)
    ax1.set_title('Total Sales by Product', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Product', fontsize=11)
    ax1.set_ylabel('Total Sales (₹)', fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    
    # Quantity sold by product
    ax2.bar(product_stats['product'], product_stats['quantity_sold'], color=colors)
    ax2.set_title('Quantity Sold by Product', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Product', fontsize=11)
    ax2.set_ylabel('Units Sold', fontsize=11)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'product_performance.png'), dpi=100)
    plt.close()
    
    # 4. Regional Sales Distribution (Pie + Bar)
    region_sales = df_full.groupby('region')['amount'].sum().reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    ax1.pie(region_sales['amount'], labels=region_sales['region'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Regional Sales Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(region_sales['region'], region_sales['amount'], color=colors)
    ax2.set_title('Sales by Region', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Region', fontsize=11)
    ax2.set_ylabel('Total Sales (₹)', fontsize=11)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'₹{height:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'regional_analysis.png'), dpi=100)
    plt.close()
    
    # 5. Sales Heatmap by Day of Week and Product
    df_full['day_of_week'] = df_full['date'].dt.day_name()
    heatmap_data = df_full.pivot_table(
        values='amount', 
        index='product', 
        columns='day_of_week', 
        aggfunc='sum', 
        fill_value=0
    )
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(columns=[d for d in day_order if d in heatmap_data.columns])
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlGnBu', 
                linewidths=0.5, cbar_kws={'label': 'Sales (₹)'})
    plt.title('Sales Heatmap: Product vs Day of Week', fontsize=16, fontweight='bold')
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Product', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'sales_heatmap.png'), dpi=100)
    plt.close()
    
    # 6. Customer Purchase Frequency Distribution
    customer_freq = df_full.groupby('customer_name').size().reset_index(name='purchases')
    
    plt.figure(figsize=(10, 5))
    plt.hist(customer_freq['purchases'], bins=range(1, customer_freq['purchases'].max()+2), 
             color='#7c3aed', edgecolor='white', alpha=0.8)
    plt.title('Customer Purchase Frequency Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Purchases', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'purchase_frequency.png'), dpi=100)
    plt.close()
    
    # 7. Average Order Value by Region
    region_aov = df_full.groupby('region')['amount'].mean().reset_index()
    region_aov = region_aov.sort_values('amount', ascending=False)
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(region_aov['region'], region_aov['amount'], color=colors)
    plt.title('Average Order Value by Region', fontsize=16, fontweight='bold')
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Average Order Value (₹)', fontsize=12)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'₹{height:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'aov_by_region.png'), dpi=100)
    plt.close()
    
    # 8. Daily Sales with Moving Average
    daily_sales = df_full.groupby('date')['amount'].sum().reset_index()
    daily_sales = daily_sales.sort_values('date')
    daily_sales['moving_avg'] = daily_sales['amount'].rolling(window=3, min_periods=1).mean()
    
    plt.figure(figsize=(12, 5))
    plt.plot(daily_sales['date'], daily_sales['amount'], 
             marker='o', label='Daily Sales', color='#4f46e5', alpha=0.6)
    plt.plot(daily_sales['date'], daily_sales['moving_avg'], 
             linewidth=3, label='3-Day Moving Average', color='#ef4444')
    plt.title('Daily Sales with Moving Average', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales (₹)', fontsize=12)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'daily_sales_ma.png'), dpi=100)
    plt.close()
    
    print("\n" + "="*60)
    print("ADDITIONAL GRAPHS GENERATED SUCCESSFULLY")
    print("="*60)
    print("✓ Monthly Sales Trend")
    print("✓ Top 5 Customers by Revenue")
    print("✓ Product Performance Comparison")
    print("✓ Regional Sales Distribution")
    print("✓ Sales Heatmap (Product vs Day of Week)")
    print("✓ Customer Purchase Frequency Distribution")
    print("✓ Average Order Value by Region")
    print("✓ Daily Sales with Moving Average")
    print("="*60 + "\n")
    
    return True
