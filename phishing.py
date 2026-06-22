import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("emails.csv")

cv = CountVectorizer()
X = cv.fit_transform(data["text"])

model = MultinomialNB()
model.fit(X, data["label"])

email = input("Enter email text: ")

test = cv.transform([email])
result = model.predict(test)

print("Result:", result[0])