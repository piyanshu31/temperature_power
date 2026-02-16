import pandas as pd               # Data manipulation and analysis
import numpy as np                # Numerical computing
import matplotlib.pyplot as plt   # Data visualization
import seaborn as sns             # Advanced data visualization
from sklearn.model_selection import train_test_split  # Splits data into train and test sets
from sklearn.linear_model import LassoCV             # LASSO regression with cross-validation
from sklearn.metrics import mean_squared_error, r2_score  # Evaluates model performance
from sklearn.preprocessing import StandardScaler, PolynomialFeatures # Standardizes and creates polynomial features
import joblib                                       # Saves and loads trained models
# Upload the Excel file
from google.colab import files
uploaded = files.upload()

# Load the Excel file into a DataFrame
file_name = list(uploaded.keys())[0]
data = pd.read_excel(file_name)
# Rename columns for easier handling (if necessary)
data.columns = [
    "Time","Num_Cotton_Wicks", "Radiation","Ambient_Temp", "Humidity",
     "Avg_Temp_With_Cooling", "Power_with_Cooling"
]

# Separate input (X) and output (Y)
X = data[["Time","Num_Cotton_Wicks", "Radiation", "Ambient_Temp", "Humidity","Avg_Temp_With_Cooling"]]
Y = data["Power_with_Cooling"]
# Correlation heatmap
correlation = data.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
plt.show()

# Feature engineering: adding polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# LASSO model with cross-validation to find optimal alpha
lasso_model = LassoCV(cv=5, alphas=np.logspace(-4, 1, 50), random_state=42)
lasso_model.fit(X_train_scaled, Y_train)

# Model predictions
Y_pred = lasso_model.predict(X_test_scaled)


# Model evaluation
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Optimal Alpha: {lasso_model.alpha_}")
print(f"Mean Squared Error: {mse}")
print(f"R-Squared: {r2}")

# Scatter plot: Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred, color='blue', alpha=0.6, label="Predicted vs Actual")
perfect_line = np.linspace(min(Y_test), max(Y_test), 100)
plt.plot(perfect_line, perfect_line, color='red', linestyle='--', label="Perfect Prediction")
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values (Y_test)")
plt.ylabel("Predicted Values (Y_pred)")
plt.legend()
plt.grid()
plt.show()


# Save the trained model
joblib.dump(lasso_model, 'lasso_power_model.pkl')

# Get feature importance from LASSO model
importance = np.abs(lasso_model.coef_)
feature_names = poly.get_feature_names_out(X.columns)

# Sort features by importance
sorted_indices = np.argsort(importance)[::-1]
sorted_features = feature_names[sorted_indices]
sorted_importance = importance[sorted_indices]

# Filter only the base features (non-polynomial)
base_features = ["Time", "Num_Cotton_Wicks", "Radiation", "Ambient_Temp", "Humidity", "Avg_Temp_With_Cooling"]
filtered_features = []
filtered_importance = []

for feature, importance_value in zip(sorted_features, sorted_importance):
    if feature in base_features and importance_value > 0:
        filtered_features.append(feature)
        filtered_importance.append(importance_value)

plt.figure(figsize=(10, 6))
plt.barh(filtered_features, filtered_importance, color='steelblue')
plt.title("Feature Importance (Base Features Only)")
plt.xlabel("Coefficient Value")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# Line graph for Power vs Number of Cotton Wicks
plt.figure(figsize=(8, 6))
data_grouped = data.groupby('Num_Cotton_Wicks')['Power_with_Cooling'].mean().sort_index()
plt.plot(data_grouped.index, data_grouped.values, marker='o', linestyle='-', color='b')
plt.title('Power Output vs Number of Cotton Wicks')
plt.xlabel('Number of Cotton Wicks')
plt.ylabel('Average Power with Cooling')
plt.grid(True)
plt.show()


# Line graph for Avg Temp with Cooling vs Number of Cotton Wicks
plt.figure(figsize=(8, 6))
data_temp_grouped = data.groupby('Num_Cotton_Wicks')['Avg_Temp_With_Cooling'].mean().sort_index()
plt.plot(data_temp_grouped.index, data_temp_grouped.values, marker='o', linestyle='-', color='g')
plt.title('Average Temperature with Cooling vs Number of Cotton Wicks')
plt.xlabel('Number of Cotton Wicks')
plt.ylabel('Average Temperature with Cooling')
plt.grid(True)
plt.show()

3# New data prediction
new_data = pd.DataFrame([{
    "Time": float(input("Enter the time (e.g., 8.0): ")),
    "Num_Cotton_Wicks": int(input("Enter the number of cotton wicks: ")),
    "Radiation": float(input("Enter the solar radiation (e.g., 500): ")),
    "Ambient_Temp": float(input("Enter the ambient temperature (e.g., 35): ")),
    "Humidity": float(input("Enter the humidity percentage (e.g., 30): ")),
    "Avg_Temp_With_Cooling": float(input("Enter the average temperature with cooling (e.g., 45): "))
}])

new_data_poly = poly.transform(new_data)
new_data_scaled = scaler.transform(new_data_poly)
predicted_power = lasso_model.predict(new_data_scaled)
print(f"\nPredicted Power_With_Cooling: {predicted_power[0]} W")

# Predict the power with cooling
predicted_power = lasso_model.predict(new_data_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(data["Num_Cotton_Wicks"], data["Power_with_Cooling"], color='b', alpha=0.6)

m, c = np.polyfit(data["Num_Cotton_Wicks"], data["Power_with_Cooling"], 1)
plt.plot(data["Num_Cotton_Wicks"], m * data["Num_Cotton_Wicks"] + c, color='r', label=f'Fit Line: y = {m:.2f}x + {c:.2f}')

plt.title("Relationship Between Number of Cotton Wicks and Power With Cooling")
plt.xlabel("Number of Cotton Wicks")
plt.ylabel("Power With Cooling")
plt.legend()
plt.grid(True)
plt.show()

# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot to show the relationship between two specific parameters
def plot_relationship(x_param, y_param):
    plt.figure(figsize=(8, 6))
    plt.scatter(data['Avg_Temp_With_Cooling'], data['Power_with_Cooling'], alpha=0.7, color='b')
    plt.title(f'Relationship Between {x_param} and {y_param}', fontsize=14)
    plt.xlabel(x_param, fontsize=12)
    plt.ylabel(y_param, fontsize=12)
    plt.grid(True)
    plt.show()

# Example: Relationship between Radiation and Power_With_Cooling
plot_relationship("Avg_Temp_With_Cooling", "Power_with_Cooling")
sns.pairplot(data)
plt.show()
