import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
import nltk
from langdetect import detect

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Function to detect the language of a text
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Function to clean text (remove special characters, punctuation, etc.)
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Load your dataset (assuming your reviews are in a CSV file with a column 'Reviews')
try:
    reviews = pd.read_csv('reviews.csv', encoding='latin-1')  # Try 'ISO-8859-1' if needed
except Exception as e:
    print(f"Error loading the CSV file: {e}")

# Step 0: Drop specified columns
reviews = reviews.drop(columns=['ID', 'Client_Name', 'Date'], errors='ignore')

# Step 1: Remove rows with "No review found" in the 'Reviews' column
reviews = reviews[reviews['Reviews'] != "No review found"]

# Step 2: Detect the language of each review
reviews['language'] = reviews['Reviews'].apply(detect_language)

# Step 3: Clean the reviews (removes special characters, punctuation, etc.)
reviews['cleaned_reviews'] = reviews['Reviews'].apply(clean_text)

# Step 4: Save the cleaned and processed dataset to a new CSV file
cleaned_dataset = reviews[['cleaned_reviews']].rename(columns={'cleaned_reviews': 'c_reviews'})
cleaned_dataset.to_csv('cleaned_reviews.csv', index=False)

print("Cleaned dataset has been saved to 'cleaned_reviews.csv'")
