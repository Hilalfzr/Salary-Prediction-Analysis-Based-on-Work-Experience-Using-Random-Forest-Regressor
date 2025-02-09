# Salary-Prediction-Analysis-Based-on-Work-Experience-Using-Random-Forest-Regressor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from google.colab import drive

drive.mount('/content/drive')
file_path = "/content/drive/My Drive/Colab Notebooks/salary_data.csv"

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    print(df.describe())

  ##cek Missing Values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing Values Ditemukan:\n", missing_values[missing_values > 0])
    else:
        print("\nTidak ada missing values.")

  ##cek Duplikasi Data
    duplicate_rows = df.duplicated().sum()
    print(f"\nDuplikasi Data: {duplicate_rows} baris ditemukan")
    if duplicate_rows > 0:
        df = df.drop_duplicates()
        print("Duplikasi telah dihapus.")
    return df

df = load_and_clean_data(file_path)

plt.figure(figsize=(10, 5))

#scatter plot pengalaman vs gaji
plt.subplot(1, 2, 1)
sns.scatterplot(x=df['experience_years'], y=df['salary'], color='blue')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")

#histogram distribusi gaji
plt.subplot(1, 2, 2)
sns.histplot(df['salary'], bins=10, kde=True, color='green')
plt.xlabel("Salary")
plt.title("Salary Distribution")

plt.tight_layout()
plt.show()

X = df[['experience_years']]
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Train Set:',X_train.shape, y_train.shape)
print('Test Set:',X_test.shape, y_test.shape)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nHasil Evaluasi Model Random Forest Regressor:")
print(f"🔹 MAE  : {mae:.2f}")
print(f"🔹 MSE  : {mse:.2f}")
print(f"🔹 RMSE : {rmse:.2f}")
print(f"🔹 R² Score : {r2:.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='red')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()
