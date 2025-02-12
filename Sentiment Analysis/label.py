import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# Load the dataset
df = pd.read_csv('cleaned_reviews.csv')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Preprocess reviews
def preprocess_review(review):
    # Remove special characters and make lowercase
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    return review.lower()

# Classify sentiment
def classify_review(review):
    score = analyzer.polarity_scores(review)
    # Adjust thresholds as necessary
    if score['compound'] > 0.05:  # Adjust positive threshold
        return 1  # Positive
    elif score['compound'] < -0.05:  # Adjust negative threshold
        return -1  # Negative
    else:
        return 0  # Neutral

# Apply preprocessing and classification
df['c_reviews'] = df['c_reviews'].apply(preprocess_review)  # Use only c_reviews
df['label'] = df['c_reviews'].apply(classify_review)

# Save to new CSV
df[['c_reviews', 'label']].to_csv('labeled_reviews.csv', index=False)

print("Labeled data saved to 'labeled_reviews.csv'")
