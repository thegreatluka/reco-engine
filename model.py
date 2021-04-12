import pandas as pd

# Updated path to raw files in github repository as Heroku wasn't able to find local files
reco_df_load = pd.read_pickle(r"user_final_rating_df.pkl")
sent_df_load = pd.read_pickle(r"items_by_sentiment_score.pkl")

def predict(user_name):

    # Top 20 Products using the Recommender System
    output = reco_df_load.loc[user_name].sort_values(ascending=False)[0:20]

    # Sort Top 5 using the Sentiment Model
    sorted_output = sent_df_load.loc[output.index.tolist()].sort_values(by='Pos%', ascending=False)[0:5]

    return sorted_output