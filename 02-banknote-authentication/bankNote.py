import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

df = pd.read_csv(url, header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

X = df[['variance', 'skewness', 'curtosis', 'entropy']]
y = df['class']


model=KNeighborsClassifier(n_neighbors=5)
predictions = model.predict()

score = cross_val_score(model , X, y, cv=5)
print("Score: ",score)
print("Average: ",score.mean())

#print(classification_report(y_test, predictions))
#print("Predictions: ",predictions)
#print("Actual: ",y_test)
