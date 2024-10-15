import pandas as pd
import matplotlib.pyplot as plt

# Specify the relative file path
file_path = "./color_differences.xlsx"

# Read the Excel file
df = pd.read_excel(file_path, sheet_name=0)

# Extract the 'Distance' column
distances = df['Distance']

# Calculate the minimum and maximum values
distance_min = distances.min()
distance_max = distances.max()

print(f"The actual range of Distance is [{distance_min}, {distance_max}]")

# Calculate quartiles
q25 = distances.quantile(0.25)
q50 = distances.median()
q75 = distances.quantile(0.75)

print(f"The interquartile range of Distance is [{q25}, {q75}], and the median is {q50}")

# Create a histogram
plt.hist(distances, bins=30, alpha=0.5, color='g', label='Distance values')

# Mark the actual range and quartile range on the plot
plt.axvline(distance_min, color='r', linestyle='dashed', linewidth=1, label='Min/Max')
plt.axvline(distance_max, color='r', linestyle='dashed', linewidth=1)
plt.axvline(q25, color='b', linestyle='dashed', linewidth=1, label='Q25/Q75')
plt.axvline(q75, color='b', linestyle='dashed', linewidth=1)

# Add legend and title
plt.legend(loc='upper right')
plt.title('Distribution of Distance values')
plt.xlabel('Distance')
plt.ylabel('Frequency')

# Display the plot
plt.show()
