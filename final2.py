import pandas as pd               # Data manipulation and analysis
import numpy as np                # Numerical computing
import matplotlib.pyplot as plt   # Data visualization
import seaborn as sns             # Advanced data visualization
from sklearn.model_selection import train_test_split  # Splits data into train and test sets
from sklearn.linear_model import LassoCV             # LASSO regression with cross-validation
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error # Evaluates model performance
from sklearn.preprocessing import StandardScaler, PolynomialFeatures # Standardizes and creates polynomial features
import joblib                                       # Saves and loads trained models
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Upload the Excel file
from google.colab import files
uploaded = files.upload()

# Load the Excel file into a DataFrame
file_name = list(uploaded.keys())[0]
data = pd.read_excel(file_name)

# Rename columns for easier handling (if necessary)
data.columns = [
    "Time","Num_Cotton_Wicks", "Radiation","Ambient_Temp", "Humidity",
     "Avg_Temp_With_Cooling", "Efficiency_with_Cooling"
]

# Separate input (X) and output (Y)
X = data[["Time","Num_Cotton_Wicks", "Radiation", "Ambient_Temp", "Humidity","Avg_Temp_With_Cooling"]]
Y = data["Efficiency_with_Cooling"]

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
rmse=root_mean_squared_error(Y_test,Y_pred)
mae=mean_absolute_error(Y_test,Y_pred)
mape=mean_absolute_percentage_error(Y_test,Y_pred)
print(f"Optimal Alpha: {lasso_model.alpha_}")
print(f"Mean Squared Error: {mse}")
print(f"R-Squared: {r2}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absoulte Error: {mae}")
print(f"Mean Abosulte Percentage Error: {mape}")

# Select base features only for clarity
base_features = ["Time", "Num_Cotton_Wicks", "Radiation", "Ambient_Temp", "Humidity", "Avg_Temp_With_Cooling"]
X_base = data[base_features]

# Standardize
scaler_base = StandardScaler()
X_scaled_base = scaler_base.fit_transform(X_base)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled_base)
components = pca.components_

# Create correlation circle
plt.figure(figsize=(8, 8))
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='black', linestyle='--')
plt.gca().add_patch(circle)

for i, feature in enumerate(base_features):
    x = components[0, i]
    y = components[1, i]
    plt.arrow(0, 0, x, y, color='r', alpha=0.6, head_width=0.03, length_includes_head=True)
    plt.text(x * 1.15, y * 1.15, feature, fontsize=10, ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA Correlation Circle (Base Features)")
plt.axis('equal')
plt.grid(True)
plt.show()

# Save the trained model
joblib.dump(lasso_model, 'lasso_efficiency_model.pkl')

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

# Line graph for Efficiency vs Number of Cotton Wicks
plt.figure(figsize=(8, 6))
data_grouped = data.groupby('Num_Cotton_Wicks')['Efficiency_with_Cooling'].mean().sort_index()
plt.plot(data_grouped.index, data_grouped.values, marker='o', linestyle='-', color='b')
plt.title('Efficiency Output vs Number of Cotton Wicks')
plt.xlabel('Number of Cotton Wicks')
plt.ylabel('Average Efficiency with Cooling')
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

# New data prediction
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
predicted_efficiency = lasso_model.predict(new_data_scaled)
print(f"\nPredicted Efficiency_With_Cooling: {predicted_efficiency[0]} %")

# Predict the efficiency with cooling
predicted_efficiency = lasso_model.predict(new_data_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(data["Num_Cotton_Wicks"], data["Efficiency_with_Cooling"], color='b', alpha=0.6)

m, c = np.polyfit(data["Num_Cotton_Wicks"], data["Efficiency_with_Cooling"], 1)
plt.plot(data["Num_Cotton_Wicks"], m * data["Num_Cotton_Wicks"] + c, color='r', label=f'Fit Line: y = {m:.2f}x + {c:.2f}')

plt.title("Relationship Between Number of Cotton Wicks and Efficiency With Cooling")
plt.xlabel("Number of Cotton Wicks")
plt.ylabel("Efficiency With Cooling")
plt.legend()
plt.grid(True)
plt.show()

# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot to show the relationship between two specific parameters
def plot_relationship(x_param, y_param):
    plt.figure(figsize=(8, 6))
    plt.scatter(data['Avg_Temp_With_Cooling'], data['Efficiency_with_Cooling'], alpha=0.7, color='b')
    plt.title(f'Relationship Between {x_param} and {y_param}', fontsize=14)
    plt.xlabel(x_param, fontsize=12)
    plt.ylabel(y_param, fontsize=12)
    plt.grid(True)
    plt.show()

# Example: Relationship between Radiation and Efficiency_With_Cooling
plot_relationship("Avg_Temp_With_Cooling", "Efficiency_with_Cooling")
sns.pairplot(data)
plt.show()
