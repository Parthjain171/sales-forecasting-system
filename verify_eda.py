import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.seasonal import seasonal_decompose

os.makedirs('charts', exist_ok=True)
plt.style.use('seaborn-v0_8-darkgrid')
COLOR_PALETTE = ["#6366f1", "#a855f7", "#ec4899", "#f43f5e", "#fbbf24"]
sns.set_palette(COLOR_PALETTE)

df = pd.read_csv("data/cleaned_data.csv", parse_dates=['Date'])

# Cap outliers at 99th percentile for cleaner visualization
cap_value = df['Total'].quantile(0.99)
df['Total'] = df['Total'].clip(upper=cap_value)
print(f"Capping outliers at 99th percentile: ${cap_value:,.0f}")


# Chart 1
plt.figure(figsize=(14, 6))
for state in df['State'].unique():
    state_data = df[df['State'] == state]
    plt.plot(state_data['Date'], state_data['Total'], label=state, alpha=0.6)
plt.title('EDA Chart 1: Sales Trend Over Time by State', fontsize=20, pad=20)
plt.xlabel('Date')
plt.ylabel('Total Sales ($)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='x-small')
plt.tight_layout()
plt.savefig('charts/eda_chart_1_sales_trend.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 2
state_totals = df.groupby('State')['Total'].sum().sort_values(ascending=True)
plt.figure(figsize=(14, 12))
bars = plt.barh(state_totals.index, state_totals.values, color='#a855f7')
for bar in bars:
    width = bar.get_width()
    label_x_pos = width + (width * 0.01)
    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'${width:,.0f}', va='center', fontsize=9)
plt.title('EDA Chart 2: Total Sales Per State', fontsize=20, pad=20)
plt.xlabel('Total Sales ($)')
plt.ylabel('State')
plt.tight_layout()
plt.savefig('charts/eda_chart_2_total_sales_per_state.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 3
df['Month'] = df['Date'].dt.month
monthly_avg = df.groupby('Month')['Total'].mean().reset_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.figure(figsize=(14, 6))
plt.plot(monthly_avg['Month'], monthly_avg['Total'], marker='o', linewidth=3, markersize=10, color='#ec4899')
plt.title('EDA Chart 3: Monthly Average Sales', fontsize=20, pad=20)
plt.xlabel('Month')
plt.ylabel('Average Sales ($)')
plt.xticks(range(1, 13), month_names)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('charts/eda_chart_3_monthly_average.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 4
top_states = df.groupby('State')['Total'].sum().sort_values(ascending=False).index[:5]
plt.figure(figsize=(14, 6))
sns.boxplot(data=df[df['State'].isin(top_states)], x='State', y='Total', palette='viridis')
plt.title('EDA Chart 4: Sales Distribution Per State', fontsize=20, pad=20)
plt.xlabel('State')
plt.ylabel('Total Sales ($) - Log Scale')
plt.yscale('log')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.savefig('charts/eda_chart_4_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print("All charts generated successfully.")
