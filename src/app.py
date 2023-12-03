import pandas as pd 
import sqlite3
import json
from pickle import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from utils import db_connect
engine = db_connect()

# your code here
movies=pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv")
credits= pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_credits.csv")

conn = sqlite3.connect("../data/movies_database.db")

movies.to_sql("movies_table",conn, if_exists="replace",index = False)
credits.to_sql("credits_table", conn , if_exists= "replace", index=False)
# merging tables 
query = """
    SELECT *
    FROM movies_table
    INNER JOIN credits_table
    ON movies_table.title = credits_table.title;
"""

total_data = pd.read_sql_query(query, conn)
conn.close()

total_data = total_data.loc[:, ~total_data.columns.duplicated()]
# data transform 
def load_json_safe(json_str,default_value=None):
    try:
        return json.loads(json_str)
    except(TypeError, json.JSONDecodeError):
        return default_value


total_data["genres"]= total_data["genres"].apply(lambda x:[item["name"] for item in json.loads(x)]if pd.notna(x) else None)
total_data["keywords"] = total_data["keywords"].apply(lambda x:[item["name"] for item in json.loads(x)] if pd.notna(x) else None)

total_data["cast"] = total_data["cast"].apply(lambda x: [item["name"] for item in json.loads(x)][:3] if pd.notna(x)else None)

total_data["crew"] = total_data["crew"].apply(lambda x:" ".join([crew_member['name'] for crew_member in load_json_safe (x)if crew_member['job'] == 'Director']))

total_data["overview"] = total_data["overview"].apply(lambda x:[x])

# creating tags 

total_data["overview"]=total_data["overview"].apply(lambda x: [str(x)])
total_data["genres"]= total_data["genres"].apply(lambda x: [str(genre) for genre in x])
total_data["keywords"]=total_data["keywords"].apply(lambda x: [str(keyword) for keyword in x ])
total_data["cast"]=total_data["cast"].apply(lambda x:[str(actor) for actor in x])
total_data["crew"]=total_data["crew"].apply(lambda x:[str(crew_member) for crew_member in x])

total_data["tags"]= total_data["overview"]+ total_data["genres"] + total_data["keywords"]+total_data["cast"]+total_data["crew"]
total_data["tags"]= total_data["tags"].apply(lambda x: ",".join(x).replace(",", " "))

total_data.drop(columns= ["genres", "keywords","cast","crew","overview"], inplace=True)

total_data.iloc[0].tags

# KNN modeling
vectorizer=TfidfVectorizer()
tfidf_martix = vectorizer.fit_transform(total_data["tags"])

model= NearestNeighbors(n_neighbors=6, algorithm="brute" , metric="cosine")
model.fit(tfidf_martix)

def get_movie_recommendations(movie_title):
    movie_index = total_data[total_data["title"] == movie_title].index[0]
    distances , indices= model.kneighbors(tfidf_martix[movie_index])
    similar_movies=[(total_data["title"][i],distances[0][j]) for j, i in enumerate(indices[0])]
    return similar_movies[1:]

input_movies = "How to Train Your Dragon"
recommendations= get_movie_recommendations(input_movies)
print("Film recommendations '{}'".format(input_movies))
for movie, distance in recommendations:
    print("- Film: {}".format(movie))

dump(model, open("../models/knn_neighbors-6_algorithm-brute_metric-cosine.sav", "wb"))
