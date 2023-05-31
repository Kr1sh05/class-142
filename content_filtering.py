from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

df = pd.read_csv("final.csv")

df = df[df['soup'].notna()]

count = CountVectorizer(stop_words='english')
count_metrics = count.fit_transform(df['soup'   ])

cosine_sim = cosine_similarity(count_metrics,count_metrics)

df = df.reset_index()
indices = pd.Series(df.index,index = df['original_title'])

def get_recommendations(original_title):
    idx = indices[original_title]
    sim_score = list(enumerate(cosine_sim[idx]))
    sim_score = sorted(sim_score,key = lambda x:x[1],reverse=True)
    sim_score = sim_score[1:11]
    movie_indices = [i[0]for i in sim_score]
    return df[['original_title','vote_count','vote_average','poster_link']].iloc[movie_indices].values.tolist()




