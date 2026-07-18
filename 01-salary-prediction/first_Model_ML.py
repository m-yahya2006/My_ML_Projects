import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/college-majors/recent-grads.csv"

df = pd.read_csv(url)

df=df.dropna()
#print(df)
#print(df.columns)

df = pd.get_dummies(df, columns=['Major_category'])
major_category = [col for col in df.columns if col.startswith('Major_category_')]
X = df[['Full_time','Part_time','Unemployment_rate','College_jobs', 'Non_college_jobs','Low_wage_jobs', *major_category]]
y = df['Median']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("r2 SCORE",r2_score(y_test, predictions))

print(predictions)

print(y_test)


