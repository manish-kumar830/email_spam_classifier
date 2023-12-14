import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('models/vectorizer.pkl','rb'))
model = pickle.load(open('models/model.pkl','rb'))

def model_predict(email):
    if email == "":
        return ""
    # 1. preprocess
    transformed_email = transform_text(email)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_email])
    # 3. predict
    prediction = model.predict(vector_input)[0]
    prediction = 1 if prediction == 1 else -1
    return prediction
