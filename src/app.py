import pandas as pd 
import numpy as np
from pickle import dump
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB-
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from utils import db_connect

engine = db_connect()

# your code here
total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv")

def apply_preprocess(df):
    df=df.drop("package_name", axis=1)
    df["review"]= df["review"].str.strip().str.lower()
    
    return df


total_data=apply_preprocess(total_data)


x=total_data["review"]
y=total_data["polarity"]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)

vec_model = CountVectorizer(stop_words="english")
x_train= vec_model.fit_transform(x_train).toarray()
x_test=vec_model.transform(x_test).toarray()

model=MultinomialNB()
model.fit(x_train, y_train)

y_pred= model.predict(x_test)

y_pred

accuracy_score(y_test,y_pred)

for model_aux in [GaussianNB(), BernoulliNB()]:
    model_aux.fit(x_train,y_train)
    y_pred_aux = model_aux.predict(x_test)
    print(f"{model_aux}with accuracy: { accuracy_score(y_test,y_pred_aux)}")

hyperparams={
    "alpha": np.linspace(0.01,10.0,200),
    "fit_prior":[True,False]
}


random_search=RandomizedSearchCV(model, hyperparams, n_iter=50, scoring="accuracy",cv=5 ,random_state=42)

random_search

random_search.fit(x_train,y_train)

print(f"Best Hyperparameters: {random_search.best_params_}")

model = MultinomialNB(alpha=1.917638190954774, fit_prior=False)
model.fit(x_train,y_train)
model.fit(x_train,y_train)
y_pred= model.predict(x_test)

accuracy_score(y_test,y_pred)


# re training the model


dump(model, open("/workspaces/machine-learning-python-template-ds-Julio/models/naive_bayes_alpha_1-9176382_fit_prior_False_42.sav", "wb"))


