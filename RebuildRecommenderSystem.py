import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

'''
Run this file to create/update the User Based Recommendation Dataframe (user_final_rating_df.pkl) 
using the whole dataset(sample30.csv)
'''

ratings = pd.read_csv(r'sample30.csv')

ratings = ratings.dropna(subset=['reviews_username'])

ratings = ratings[['reviews_username','name','reviews_rating','id']]

# Handle multiple reviews by same user
ratings_filtered = ratings.groupby(['reviews_username', 'name']).agg({'reviews_rating':['mean']})
ratings_filtered.columns = ratings_filtered.columns.map('_'.join)
ratings_filtered = ratings_filtered.reset_index()

# Copy the train dataset into dummy_train
dummy_train = ratings_filtered.copy()

# The products not rated by user is marked as 1 for prediction.
dummy_train['reviews_rating_mean'] = dummy_train['reviews_rating_mean'].apply(lambda x: 0 if x>=1 else 1)

# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot(
    index='reviews_username',
    columns='name',
    values='reviews_rating_mean'
).fillna(1)

# Create a user-product matrix.
df_pivot = ratings_filtered.pivot(
    index='reviews_username',
    columns='name',
    values='reviews_rating_mean'
)

mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T

# Adjusted Cosine
# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)

user_correlation[user_correlation<0]=0

user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))

user_final_rating = np.multiply(user_predicted_ratings,dummy_train)

pd.to_pickle(user_final_rating, r'user_final_rating_df.pkl')