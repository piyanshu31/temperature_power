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
