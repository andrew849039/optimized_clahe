import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the Excel file using a relative path
df = pd.read_excel('./excel_a_without_duplicates.xlsx')

# Calculate quartiles
quantiles = df[['L', 'a', 'b']].quantile([0.25, 0.5, 0.75]).values

# Plot for each color channel
for i, color in enumerate(['L', 'a', 'b']):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(df[color], bins=30, alpha=0.5)
    ax.axvline(x=quantiles[0, i], color='r', linestyle='--', label='Q1')
    ax.axvline(x=quantiles[1, i], color='g', linestyle='--', label='Q2 (Median)')
    ax.axvline(x=quantiles[2, i], color='b', linestyle='--', label='Q3')
    ax.text(quantiles[0, i], 0, f'Q1={quantiles[0, i]:.2f}', horizontalalignment='right', verticalalignment='bottom', color='r')
    ax.text(quantiles[1, i], 0, f'Q2={quantiles[1, i]:.2f}', horizontalalignment='right', verticalalignment='bottom', color='g')
    ax.text(quantiles[2, i], 0, f'Q3={quantiles[2, i]:.2f}', horizontalalignment='right', verticalalignment='bottom', color='b')
    ax.legend()
    ax.set_title(f'{color} channel distribution')
    ax.set_xlabel('Color value')
    ax.set_ylabel('Frequency')
    plt.show()
