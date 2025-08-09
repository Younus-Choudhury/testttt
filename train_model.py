# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the data from the CSV file
df = pd.read_csv('insurance.csv')

# Preprocess the data to convert categorical features into numerical ones
# This is required for the machine learning model to understand the data
le_sex = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'])

le_smoker = LabelEncoder()
df['smoker'] = le_smoker.fit_transform(df['smoker'])

le_region = LabelEncoder()
df['region'] = le_region.fit_transform(df['region'])

# Define the features (X) and the target (y)
# X contains all columns except 'charges'
# y is the 'charges' column, which is what we want to predict
X = df.drop('charges', axis=1)
y = df['charges']

# Split the data into training and testing sets
# We use a 80/20 split, so 80% of the data is for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor model
# This model is a good choice for regression tasks and is often used for this kind of data
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using the training data
model.fit(X_train, y_train)

# Save the trained model to a file using joblib
# This creates the 'insurance_rf_model.pkl' file that the Flask app uses
joblib.dump(model, 'insurance_rf_model.pkl')

print("Model training complete and saved as 'insurance_rf_model.pkl'")