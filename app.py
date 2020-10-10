import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB as mn
from sklearn.model_selection import train_test_split as tts
import streamlit as st


data = pd.read_csv('movie_review.csv', delimiter = ',')
reviews = data['text']
y = data['tag']

def clean(data):
    
    clean_data = []
    punc = list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
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

a = clean(reviews)
x1, x2, y1, y2 = tts(a, y, test_size = 0.1)

cv = TfidfVectorizer()
X = cv.fit_transform(x2).todense()
clf = mn()
clf.fit(X, y2)
keys = {'pos' : 'It is a positive review :)', 'neg': 'It is a negative review :('}
st.title("Movie Review Classifier")
message = st.text_area("Enter Text")
test_val = cv.transform([message])
pred = clf.predict(test_val)
if st.button("Predict"):
    st.title(keys[pred[0]])
