import pandas as pd
import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Ignore warnings
warnings.filterwarnings("ignore")

# Preprocess text
def preprocess_text(text):
    # Clean and preprocess the text
    text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text).lower()
    words = text.split()
    words = [word for word in words if word not in set(stopwords.words('english'))]
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    processed_text = ' '.join(words)
    return processed_text

# Load the dataset and verify column names
df = pd.read_csv('labeled_reviews.csv')
print(df.columns)  # Verify column names

# Ensure 'c_reviews' and 'label' columns exist
if 'c_reviews' not in df.columns or 'label' not in df.columns:
    raise KeyError("Columns 'c_reviews' and 'label' not found in dataset")

# Preprocess text
df['Processed_Text'] = df['c_reviews'].apply(preprocess_text)

# Create Bag of Words model
cv = CountVectorizer(max_features=1500, ngram_range=(1, 2))
X = cv.fit_transform(df['Processed_Text']).toarray()

# Update the target variable extraction to match the new labels
y = df['label'].values

# Save CountVectorizer
joblib.dump(cv, 'cv.pkl')
print("CountVectorizer saved as 'cv.pkl'.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Initialize classifiers
classifier_nb = MultinomialNB()
classifier_rf = RandomForestClassifier()
classifier_svm = SVC()
classifier_lr = LogisticRegression()

# Train Naive Bayes
classifier_nb.fit(X_train, y_train)
y_pred_nb = classifier_nb.predict(X_test)

# Metrics for Naive Bayes
accuracy_nb = accuracy_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb, average='weighted')
precision_nb = precision_score(y_test, y_pred_nb, average='weighted')
recall_nb = recall_score(y_test, y_pred_nb, average='weighted')
print(f"\n\nNaive Bayes Accuracy: {accuracy_nb}")
print(f"Naive Bayes F1 Score: {f1_nb}")
print(f"Naive Bayes Precision: {precision_nb}")
print(f"Naive Bayes Recall: {recall_nb}")

# Train Random Forest
classifier_rf.fit(X_train, y_train)
y_pred_rf = classifier_rf.predict(X_test)

# Metrics for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
print(f"\n\nRandom Forest Accuracy: {accuracy_rf}")
print(f"Random Forest F1 Score: {f1_rf}")
print(f"Random Forest Precision: {precision_rf}")
print(f"Random Forest Recall: {recall_rf}")

# Train SVM
classifier_svm.fit(X_train, y_train)
y_pred_svm = classifier_svm.predict(X_test)

# Metrics for SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
print(f"\n\nSVM Accuracy: {accuracy_svm}")
print(f"SVM F1 Score: {f1_svm}")
print(f"SVM Precision: {precision_svm}")
print(f"SVM Recall: {recall_svm}")

# After calculating accuracies for all models, determine the best model
best_model = None
best_accuracy = 0

# Store model details for comparison
models = {
    'Naive Bayes': (classifier_nb, accuracy_nb),
    'Random Forest': (classifier_rf, accuracy_rf),
    'SVM': (classifier_svm, accuracy_svm),
}

# Find the best model based on accuracy
for model_name, (model, accuracy) in models.items():
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = model_name

# Save the best model
joblib.dump(best_model, 'model.pkl')
print(f"\n\nBest model '{best_model_name}' with accuracy {best_accuracy} saved as 'model.pkl'.")

print("\n\nUnique labels in the dataset:", df['label'].unique(),"\n\n")

# Count sentiment occurrences
sentiment_counts = df['label'].value_counts()
sentiment_labels = sentiment_counts.index
sentiment_values = sentiment_counts.values

# Set up the color mapping for the sentiments
color_mapping = {
    1: 'green',
    -1: 'red',
    0: 'yellow'
}

# Create a list of colors based on sentiment labels
colors = [color_mapping[label] for label in sentiment_labels]

# Set up the bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(sentiment_labels, sentiment_values, color=colors)

# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set x-ticks and labels
plt.xticks(sentiment_labels, ['Positive', 'Negative', 'Neutral'])  # Replace with descriptive labels
plt.title('Sentiment Distribution', fontsize=16)
plt.xlabel('Sentiment', fontsize=14)
plt.ylabel('Number of Reviews', fontsize=14)
plt.ylim(0, max(sentiment_values) + 100)

# Annotate each bar with the corresponding count
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), 
             ha='center', va='bottom', fontsize=12, color='black')

# Add a legend
handles = [plt.Rectangle((0,0),1,1, color=color_mapping[label]) for label in sentiment_labels]
labels = ['Positive', 'Negative', 'Neutral']
plt.legend(handles, labels, title='Sentiment', title_fontsize='13', fontsize='12')

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# Word Cloud Generation
# Combine all the processed text to create a single string
all_text = ' '.join(df['Processed_Text'])

# Create the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axis
plt.title('Word Cloud for All Reviews', fontsize=16)
plt.show()
