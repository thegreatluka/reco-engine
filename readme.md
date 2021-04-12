###### Sentiment Based Recommendation Engine


**Heroku URL**: https://reco-engine101.herokuapp.com/

**Important File Summary**:

**app.py** : Contains API Definitions<br/>
**model.py** : Contains API Implementations<br/>
**RebuildRecommenderSystem.py** : Run this file to create/update the User Based Recommendation Dataframe (user_final_rating_df.pkl) 
using the whole dataset(sample30.csv)<br/>
**RebuildSentimentModel.py** : Run this file to create/update the Item -> Postive Review % Mapping Dataframe (items_by_sentiment_score.pkl)
using the chosen Logistic Regression Sentiment Model and using the whole dataset(sample30.csv)<br/>
**sample30.csv** : User Product Review Datatset used by **RebuildRecommenderSystem.py** and **RebuildSentimentModel.py**<br/> 
