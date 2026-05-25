import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#dataset
data = pd.read_csv("cg.csv")

#columns
data = data[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']]

#fill missing values
data['Age'] = data['Age'].fillna(data['Age'].mean())

#convert text to number
data['Sex'] = data['Sex'].map({
    'male': 0,
    'female': 1
})

#Input and output
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

#split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

 #model
model = LogisticRegression()

model.fit(X_train, y_train)

#predict
prediction = model.predict(X_test)

#accuracy
print("Accuracy:",
      round(
          accuracy_score(y_test,prediction) * 100,2),"%"
)