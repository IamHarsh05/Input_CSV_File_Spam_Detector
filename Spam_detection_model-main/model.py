import spacy
import pandas
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


dataset = pandas.read_csv('spam.csv', encoding="ISO-8859-1")

en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words
dataset['clean'] = dataset['v2'].apply(lambda x: ' '.join(
    [word for word in x.split() if word.lower() not in (sw_spacy)]))

count_vectorizer = CountVectorizer()
count = count_vectorizer.fit_transform(dataset['v2'])
Y = dataset['v2']

arr = dataset.values
label = np.delete(arr, [1, 2, 3, 4, 5], axis=1)
label = label.ravel()

x_train, x_test, y_train, y_test = train_test_split(
    count, label, test_size=0.2, random_state=42)
logistic = LogisticRegression()
logistic.fit(x_train, y_train)

# Saving model to disk
pickle.dump(logistic, open('model.pkl', 'wb'))
