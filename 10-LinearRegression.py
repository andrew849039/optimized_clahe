import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import os

# Read data
print("Reading data...")
data = pd.read_excel('./merged_color_clusters.xlsx')
print("Data reading completed!")
print()

# Group by image to determine main and support colors
print("Determining main and support colors...")
grouped = data.groupby('Image')

distances = []
for _, group in grouped:
    main_color = group.loc[group['Ratio'].idxmax()]
    support_colors = group.loc[group['Ratio'] < 1]

    for _, support_color in support_colors.iterrows():
        distance = np.sqrt(
            (main_color['L'] - support_color['L']) ** 2 +
            (main_color['a'] - support_color['a']) ** 2 +
            (main_color['b'] - support_color['b']) ** 2
        )
        distances.append({
            'Main Color L': main_color['L'],
            'Main Color a': main_color['a'],
            'Main Color b': main_color['b'],
            'Support Color L': support_color['L'],
            'Support Color a': support_color['a'],
            'Support Color b': support_color['b'],
            'Distance': distance
        })
print("Main and support colors determined!")
print()

# Build a multiple linear regression model
print("Building a multiple linear regression model...")
distances_df = pd.DataFrame(distances)
X = distances_df[['Main Color L', 'Main Color a', 'Main Color b', 'Support Color L', 'Support Color a', 'Support Color b']]
y = distances_df['Distance']

model = LinearRegression()
model.fit(X, y)
print("Multiple linear regression model built!")
print()

# Output model parameters
print('Model parameters:')
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)
print()

# Output the integrated mathematical formula
variables = ['Main Color L', 'Main Color a', 'Main Color b', 'Support Color L', 'Support Color a', 'Support Color b']
equation = "Regression model formula: Y = {:.2f}".format(model.intercept_)
for coef, variable in zip(model.coef_, variables):
    equation += " + {:.2f}*{}".format(coef, variable)
print(equation)
print()

# Save the model to a new folder
output_dir = './output-1/'
print("Saving the model...")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
pickle.dump(model, open(os.path.join(output_dir, "color_model.pkl"), 'wb'))
print("Model saved!")
print()

# Output color difference data
print("Exporting color difference data...")
output_filepath = './color_differences.xlsx'
distances_df.to_excel(output_filepath, index=False)
print("Color difference data exported!")
