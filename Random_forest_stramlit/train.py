import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib


data = pd.read_csv("data/Iris.csv")

data1 = data.drop("Id",axis=1)



# mapping Species to numarical values
data1["Species"] = data1["Species"].map({
    "Iris-setosa":0,
    "Iris-versicolor":1,
    "Iris-virginica":2
})
print(data1.head())


# define labels and features
X = data1.drop("Species",axis=1)
Y = data1["Species"]
print(Y.head())
print(Y.value_counts())

# Spliting data
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=42)

# training the model
model = RandomForestClassifier()
model.fit(X_train,Y_train)

# prediction
predictions = model.predict(X_test)
print(predictions)

# accuracy
accuracy = accuracy_score(Y_test,predictions)
print(accuracy)

# saving trained model
joblib.dump(model,"model/model.pkl")




