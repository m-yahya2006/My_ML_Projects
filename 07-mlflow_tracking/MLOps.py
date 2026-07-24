import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


data = load_breast_cancer()
X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


for n_est , depth in [(50,3),(100,5),(200,10)]:
    with mlflow.start_run():

        mlflow.log_param("n_estimator",n_est)
        mlflow.log_param("max_depth",depth)

        model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        accuracy =  accuracy_score(predict , y_test)
        mlflow.sklearn.log_model(model,name="model")
        mlflow.log_metric("Accuracy", accuracy)
        print("Accuracy: ",accuracy)
        print("\nClassification Report: \n")
        print(classification_report(y_test, predict))