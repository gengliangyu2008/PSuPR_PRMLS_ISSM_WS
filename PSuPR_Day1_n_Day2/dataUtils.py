import pandas as pd
from sklearn.model_selection import train_test_split

# Loading the dataset
diabetes_data = pd.read_csv('diabetes.csv')

# Print the first 5 rows of the dataframe.
print(diabetes_data.head())

# observing the shape of the data
print(diabetes_data.shape)

X = diabetes_data.drop("Outcome",axis = 1)
y = diabetes_data.Outcome
print(X.head())

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)
print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)