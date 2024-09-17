import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv(r"C:\Users\reesh\Desktop\House Price Prediction\Data\train.csv")
test_data = pd.read_csv(r"C:\Users\reesh\Desktop\House Price Prediction\Data\test (2).csv")

print(train_data.info())
print(train_data.head())
print(train_data.describe())

missing_values = train_data.isnull().sum()
print("Missing values in each column:\n", missing_values[missing_values > 0])

plt.figure(figsize=(10, 6))
sns.histplot(train_data['SalePrice'], bins=30, kde=True)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

for column in train_data.select_dtypes(include=[np.number]).columns:
    train_data[column] = train_data[column].fillna(train_data[column].median())

for column in train_data.select_dtypes(include=[object]).columns:
    train_data[column] = train_data[column].fillna(train_data[column].mode()[0])

train_data = pd.get_dummies(train_data, drop_first=True)

print("Shape of train data after preprocessing:", train_data.shape)

X = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]  
y = train_data['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
print(f'Mean Squared Error: {mse}')

test_data_id = test_data['Id']  

for column in test_data.select_dtypes(include=[np.number]).columns:
    test_data[column] = test_data[column].fillna(test_data[column].median())

for column in test_data.select_dtypes(include=[object]).columns:
    test_data[column] = test_data[column].fillna(test_data[column].mode()[0])

test_data = pd.get_dummies(test_data, drop_first=True)

test_data = test_data.reindex(columns=X.columns, fill_value=0)

X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]  

test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    'Id': test_data_id, 
    'SalePrice': test_predictions
})

submission.to_csv(r"C:\Users\reesh\Desktop\House Price Prediction\Data\submissions.csv", index=False)
print("Predictions saved to submissions.csv")
