# Import necessary libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib

# Function to preprocess new input
def preprocess_text(text):
    text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text).lower()
    words = text.split()
    words = [word for word in words if word not in set(stopwords.words('english'))]
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    processed_text = ' '.join(words)
    return processed_text

# Load the model and vectorizer
classifier_nb = joblib.load("model.pkl")  # Adjust based on your saved model
cv = joblib.load("cv.pkl")  # Load the CountVectorizer if you saved it

def predict_sentiment():
    while True:
        # Prompt user for a review
        new_review = input("Enter a review (or type 'exit' to quit): ")
        if new_review.lower() == 'exit':
            print("Exiting the sentiment analysis.")
            break
        
        # Preprocess the new review
        processed_review = preprocess_text(new_review)

        # Transform the processed review using the same CountVectorizer
        X_new = cv.transform([processed_review]).toarray()

        # Make prediction
        prediction = classifier_nb.predict(X_new)

        # Output the prediction
        print(f"The predicted sentiment for the review is: {prediction[0]}")

# Call the function to start predicting
predict_sentiment()
