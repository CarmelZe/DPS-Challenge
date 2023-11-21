import pandas as pd
import matplotlib.pyplot as plt

# Specify the path to the dataset
dataset_path = 'data/monatszahlen2307_verkehrsunfaelle_10_07_23_nosum.csv'

# Load the dataset
df = pd.read_csv(dataset_path)

# Convert the 'MONAT' column to datetime format
df['Date'] = pd.to_datetime(df['MONAT'], format='%Y%m')

# Filter records up to the end of 2020
df_filtered = df[df['Date'].dt.year <= 2020]

# Visualize the historical number of accidents per category
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed

df_filtered['MONATSZAHL'].value_counts().plot(kind='barh')
plt.title('Historical Number of Accidents per Category')
plt.xlabel('Category')
plt.ylabel('Number of Accidents')

# Save the plotted image
plt.savefig('Visualization-Image/historical_accidents_per_category.png')

# Visualize the historical number of accidents per category
df_filtered['AUSPRAEGUNG'].value_counts().plot(kind='barh')
plt.title('Historical Number of Accidents per Accident Type')
plt.xlabel('Accident Type')
plt.ylabel('Number of Accidents')

# Adjust layout to prevent cropping of labels
plt.tight_layout()

# Save the plotted image
plt.savefig('Visualization-Image/historical_accidents_per_accident_type.png')