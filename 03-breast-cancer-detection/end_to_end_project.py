from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

data=load_breast_cancer()
df=pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model_2 =DecisionTree()
#model_3 = RandomForestTress()

model_1 = KNeighborsClassifier(n_neighbors=17)
model_1.fit(X_train, y_train)

predict = model_1.predict(X_test)

#print("Accuracy: ",accuracy_score(y_test, predict))
score = cross_val_score(model_1, X, y , cv=5, scoring='recall')
print(score)
print("Average: ",score.mean())

#print(classification_report(y_test, predict))


