import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
from pickle import dump
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from utils import db_connect
engine = db_connect()


# your code here
total_data= pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv")

x= total_data[["MedInc", "Latitude","Longitude"]]

# spliting data set 
x_train , x_test= train_test_split(x, test_size= 0.2, random_state=42)

# kMeans
from sklearn.cluster import KMeans

model_unsup=KMeans(n_clusters=6, n_init="auto", random_state=42)
model_unsup.fit(x_train)

# inserting clusters

y_train=list(model_unsup.labels_)
x_train["cluster"] = y_train

fig, axis = plt.subplots(1,3, figsize=(15,5))

sns.scatterplot(ax= axis[0], data= x_train, x= "Latitude", y= "Longitude", hue="cluster", palette="deep")
sns.scatterplot(ax= axis[1], data= x_train, x= "Latitude", y= "MedInc", hue="cluster", palette="deep")
sns.scatterplot(ax= axis[2], data= x_train, x= "Longitude", y= "MedInc", hue="cluster", palette="deep")

plt.tight_layout()

plt.show()

y_test = list(model_unsup.predict(x_test))
x_test["cluster"]= y_test

# ploting test results above 

fig, axis= plt.subplots(1,3, figsize =(15,5))

sns.scatterplot(ax = axis[0], data= x_train,x="Latitude",y="Longitude", hue = "cluster",palette="deep", alpha=0.1)
sns.scatterplot(ax = axis[1], data= x_train,x="Latitude",y="MedInc", hue = "cluster",palette="deep", alpha=0.1)
sns.scatterplot(ax = axis[2], data= x_train,x="Longitude",y="MedInc", hue = "cluster",palette="deep", alpha=0.1)

sns.scatterplot(ax = axis[0], data= x_test,x="Latitude",y="Longitude", hue = "cluster",palette="deep", marker="+")
sns.scatterplot(ax = axis[1], data= x_test,x="Latitude",y="MedInc", hue = "cluster",palette="deep", marker="+")
sns.scatterplot(ax = axis[2], data= x_test,x="Longitude",y="MedInc", hue = "cluster",palette="deep", marker="+")

plt.tight_layout

for ax in axis:
    ax.legend([],[], frameon=False)

plt.show()

# training a supervised clafacation model (decision Tree)
model_sup = DecisionTreeClassifier(random_state=42)

model_sup.fit(x_train,y_train)

fig= plt.figure(figsize=(15,15))

tree.plot_tree(model_sup, feature_names=list(x_train.columns), class_names=["0","1","2","3","4","5"],filled = True)

plt.show() 

y_pred=model_sup.predict(x_test)

y_pred

accuracy_score(y_test,y_pred)


dump(model_unsup, open("/workspaces/machine-learning-python-template-ds-Julio/models/k-means_default_42.sav", "wb"))
dump(model_sup, open("/workspaces/machine-learning-python-template-ds-Julio/models/decision_tree_classifier_default_42.sav", "wb"))