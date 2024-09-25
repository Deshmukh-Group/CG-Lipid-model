import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# Load the CSV file
data = pd.read_csv('clean1.csv')

# Separate parameters and properties
parameters = data.iloc[:, :18]
properties = data.iloc[:, 18:27]

# Function to calculate partial correlation
def partial_corr(x, y, covariates):
    covariates = add_constant(covariates)
    x_res = OLS(x, covariates).fit().resid
    y_res = OLS(y, covariates).fit().resid
    corr, _ = pearsonr(x_res, y_res)
    return corr

# Initialize a DataFrame to store partial correlations
partial_corr_df = pd.DataFrame(index=properties.columns, columns=parameters.columns)

# Calculate partial correlations
for prop in properties.columns:
    for param in parameters.columns:
        covariates = data.drop(columns=[param, prop]).values
        partial_corr_value = partial_corr(data[param].values, data[prop].values, covariates)
        partial_corr_df.loc[prop, param] = partial_corr_value

# Convert partial correlation DataFrame to float
partial_corr_df = partial_corr_df.astype(float)

# Create a DataFrame for annotation text, leaving empty strings for non-significant correlations
annotation_df = partial_corr_df.applymap(lambda x: f'{x:.2f}' if abs(x) > 0.2 else '')

# Plotting the heatmap
plt.figure(figsize=(20, 9))
ax = sns.heatmap(partial_corr_df, annot=annotation_df, cmap='vlag', linewidths=1.0, fmt='', annot_kws={"size": 15, "color": 'white', "weight": 'bold'}, vmin=-0.45, vmax=0.45)

x_labels = ['CHO','PO2','D3M','COH','MTF','COH-W','MTF-W','CHO-W','PO2-W','CHO','PO2','D3M','COH','MTF','COH-W','MTF-W','CHO-W','PO2-W']
y_labels = ['AL','Dpp','Kc','AL','Dpp','Kc','AL','Dpp','Kc']
ax.set_xticklabels(x_labels, fontsize=20, rotation=90)
ax.set_yticklabels(y_labels, fontsize=20, rotation=0, ha='right')

ax.axvline(x=9.0, color='black', linewidth=2.0)
ax.axhline(y=3.0, color='black', linewidth=2.0)
ax.axhline(y=6.0, color='black', linewidth=2.0)

ax.axvline(x=5.0, color='black', linestyle='--', linewidth=1.0)
ax.axvline(x=14.0, color='black', linestyle='--', linewidth=1.0)

#ax.axvline(x=0.0, color='black', linewidth=2.0)
#ax.axvline(x=18.0, color='black', linewidth=2.0)
#ax.axhline(y=0.0, color='black', linewidth=2.0)
#ax.axhline(y=9.0, color='black', linewidth=2.0)

# Customize the plot
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('')

# Make x and y ticks bold and set label size
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20, rotation=0, ha='right')

ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

# Make legend text bold and fontsize 20
colorbar = ax.collections[0].colorbar
colorbar.ax.yaxis.set_tick_params(labelsize=25, width=2, color='black')

# Adjust layout
plt.tight_layout()
plt.savefig('heatmap_partial_correlation.png', dpi=300)
