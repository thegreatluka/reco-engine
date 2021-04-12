import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

'''
Run this file to create/update the Item -> Postive Review % Mapping Dataframe (items_by_sentiment_score.pkl)
using the chosen Logistic Regression Sentiment Model and using the whole dataset(sample30.csv)
'''

# Importing dataset and removing unneeded attributes
reviews = pd.read_csv(r'sample30.csv')
reviews.drop(reviews.columns.difference(['reviews_title','reviews_text','user_sentiment','name']), 1, inplace=True)
reviews["review_text_all"] = reviews["reviews_title"].astype(str) + " " + reviews["reviews_text"]

# Punctuation Handling
reviews['review_text_all'] = reviews['review_text_all'].str.replace('[^a-zA-Z\s]','')
print('Punctuation Handling........Done!')

# Stopwords Handling
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
reviews['review_text_all_without_stopwords'] = reviews['review_text_all'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print('Stopwords Handling........Done!')

# Apply Spellchecking
from spellchecker import SpellChecker

spell = SpellChecker()
reviews['review_text_all_without_stop_corrected'] = reviews.review_text_all_without_stopwords.apply(lambda txt: ''.join(spell.correction(txt)))
print('Spellchecking........Done!')

# Apply Lowercasing
reviews['review_text_all_without_stop_corrected'] = reviews['review_text_all_without_stop_corrected'].str.lower()
print('Lowercasing........Done!')

# Apply Lemmatization
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

lmtzr = WordNetLemmatizer()

def lemmatize_text(text):
    return ' '.join([lmtzr.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)])

reviews['rev_text_lemma'] = reviews['review_text_all_without_stop_corrected'].apply(lemmatize_text)
print('Lemmatizaton........Done!')


all_text = reviews['rev_text_lemma']
train_text = reviews['rev_text_lemma']
y = reviews['user_sentiment']

# TF-IDF Vectorization (Word)
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1,1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)

# TF-IDF Vectorization (Character)
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2,6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)

train_features = hstack([train_char_features, train_word_features])
print('TF-IDF Vectorization........Done!')

# Load/Deserialize Classifier
lrc = pd.read_pickle(r"Logistic_Regression_Model.pkl")
lr_pred = lrc.predict(train_features)


reviews['sentiment'] = lr_pred
reviews['sentiment'] = reviews.apply(lambda row: 'Positive' if row['sentiment']==1 else 'Negative', axis=1)
print(reviews)

temp = reviews[['name','user_sentiment','review_text_all']]
fin = pd.pivot_table(temp, index= ['name'], columns=['user_sentiment'], aggfunc='count')
fin.columns = fin.columns.droplevel()
fin = fin.fillna(0)
fin['Pos%'] = fin.Positive / (fin.Positive + fin.Negative)
print(fin)
pd.to_pickle(fin, r'items_by_sentiment_score.pkl')