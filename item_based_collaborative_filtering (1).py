
import pandas as pd

data = pd.read_csv('songsDataset.csv',nrows=10000)

data.shape

data.head()

data.describe()

data.isnull().sum()

# data.dropna(inplace=True)

data.isnull().sum()

data.duplicated().sum()

"""Convert the dataset into a matrix where rows represent users, columns represent songs, and cells represent ratings"""

data.head()

print(data.columns)

# show rating column
data["'rating'"].head()

item_matrix = data.pivot_table(index="'userID'", columns="'songID'", values="'rating'")

"""Fill missing values with zeros"""

item_matrix.isnull().sum()

item_matrix.head()

item_matrix.fillna(0, inplace=True)

"""Compute the cosine similarity between items"""

from sklearn.metrics.pairwise import cosine_similarity

item_similarity = cosine_similarity(item_matrix.T)

# containing the similarity scores between items (songs).
item_similarity_df = pd.DataFrame(item_similarity, index=item_matrix.columns, columns=item_matrix.columns)

"""Create a function to get similar items"""

def get_similar_items(song_id, item_similarity_df, top_n=5):
    similar_scores = item_similarity_df[song_id].sort_values(ascending=False)
    similar_items = similar_scores.iloc[1:top_n+1].index
    return similar_items

def recommend_songs(user_id, item_matrix, item_similarity_df, top_n=5):

    user_ratings = item_matrix.loc[user_id]
    user_ratings = user_ratings[user_ratings > 0]

    recommendations = pd.Series(dtype=float)
    for song, rating in user_ratings.items():
        similar_items = get_similar_items(song, item_similarity_df, top_n)
        for similar_item in similar_items:
            if similar_item in recommendations:
                recommendations[similar_item] += rating
            else:
                recommendations[similar_item] = rating

    recommendations = recommendations.sort_values(ascending=False)
    return recommendations.head(top_n).index

user_id = int(input("Enter UserID : "))
recommended_songs = recommend_songs(user_id, item_matrix, item_similarity_df)
print("Recommended songs for user", user_id, ":", recommended_songs)









"""## Using sparse matrix"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare the data
data = pd.read_csv('songsDataset.csv',nrows=10000)

# Create a sparse matrix from the data
sparse_item_matrix = csr_matrix((data["'rating'"].values, (data["'userID'"].values, data["'songID'"].values)))

# Compute the cosine similarity between items
item_similarity = cosine_similarity(sparse_item_matrix.T)

# Create a function to get similar items
def get_similar_items(song_id, item_similarity, top_n=5):
    similar_scores = item_similarity[song_id].flatten()
    top_n_indices = similar_scores.argsort()[:-top_n-1:-1][1:]
    return top_n_indices

# Create a function to recommend songs
def recommend_songs(user_id, sparse_item_matrix, item_similarity, top_n=5):
    user_ratings = sparse_item_matrix[user_id].toarray().flatten()
    user_ratings = [rating for rating in user_ratings if rating > 0]
    song_ids = [i for i, rating in enumerate(user_ratings) if rating > 0]

    recommendations = {}
    for song_id in song_ids:
        similar_items = get_similar_items(song_id, item_similarity, top_n)
        for similar_item in similar_items:
            if similar_item in recommendations:
                recommendations[similar_item] += user_ratings[song_id]
            else:
                recommendations[similar_item] = user_ratings[song_id]

    recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return [song_id for song_id, _ in recommendations[:top_n]]

# Get user ID from input
user_id = int(input("Enter UserID : "))

# Make recommendations
recommended_songs = recommend_songs(user_id, sparse_item_matrix, item_similarity)
print("Recommended songs for user", user_id, ":", recommended_songs)





