import pandas as pd
import numpy as np
import string
import re
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB as mn
import streamlit as st

data = pd.read_csv('imdb.csv', delimiter = ',')
reviews = data['review']
targets = data['sentiment']

def clean(data):
    
    clean_data = []
    punc = list(string.punctuation)
    for line in data:
        words = []
        for word in line.split():
            #removing punctuations (like @ or #)
            t = [w for w in word if w not in punc]
            word = "".join(t)
            #removing numbers
            word = re.sub(r"\d+", "", word)
            #removing URLs
            word = re.sub(r'https?\S+', '', word)
            word = word.lower()
            
            if(len(word) > 2):
                words.append(word)
                
        clean_data.append(" ".join(words))
    
    return clean_data

f = open('clean', 'rb')
clean_words = pk.load(f)
f.close()

Y = np.array(targets == 'positive', dtype = np.int32)
cv = TfidfVectorizer(max_features = 50000, max_df = 0.5, ngram_range = (1, 3))
X = cv.fit_transform(clean_words).todense()
clf = mn()
clf.fit(X, Y)
st.title("Movie Review Classifier")
message = st.text_area("Enter Text","Type Here ..")
test_val = cv.transform([message])
pred = clf.predict(test_val)
if st.button("Predict"):
    st.title(keys[pred])
