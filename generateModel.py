import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle

import warnings
warnings.filterwarnings('ignore')

dataset_path = 'Data/monatszahlen2307_verkehrsunfaelle_10_07_23_nosum.csv'

df = pd.read_csv(dataset_path)

# Process the data
data = df.iloc[:, :5].copy()

data_2020 = data[data["JAHR"] <= 2020].reset_index(drop=True)
data_2020.rename(columns={"MONATSZAHL": "Category", "AUSPRAEGUNG": "Accident_type", "JAHR": "Year", "MONAT": "Month", "WERT": "Value"}, inplace=True)

data_2020.drop(data_2020[data_2020["Month"] == "Summe"].index, inplace=True)

data_2020['Month'] = pd.to_numeric(data_2020['Month'].astype(str).str[-2:], errors='coerce')

# Categorical features to encode using one-hot encoding
features = ['Category', 'Accident_type', 'Year', 'Month']

# setting input features X and target y
X = data_2020[features].copy()  # Copy the data to avoid DeprecationWarning
y = data_2020['Value'].copy()    # Copy the data to avoid ValueError

# category column encoding
cat_mapping = {'Verkehrsunfälle': 0, 'Alkoholunfälle': 1, 'Fluchtunfälle': 2}
X.loc[:, 'Category'] = X['Category'].map(cat_mapping)

# Accident_type column mapping
accid_mapping = {'insgesamt': 0, 'Verletzte und Getötete': 1, 'mit Personenschäden': 2}
X.loc[:, 'Accident_type'] = X['Accident_type'].map(accid_mapping)

# label encoding for 'Year' column
le_year = LabelEncoder()
X.loc[:, 'Year'] = le_year.fit_transform(X['Year'])

# label encoding for 'Month' column
le_month = LabelEncoder()
X.loc[:, 'Month'] = le_month.fit_transform(X['Month'])

# train and test split and building baseline model to predict target features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# Save the model in the 'Models' folder
model_filename = 'Models/model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(rf_model, f)
