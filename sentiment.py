import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import pickle
import gradio as gr
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
# Loadinf and preprocessing the dataset
file_path = 'C:\\Users\\Lenovo\\OneDrive\\Desktop\\kgpp\\kgpdata.csv'
data = pd.read_csv(file_path, header=None, names=['Combined'])

# Spliting the single column into 'Comments' and 'Sentiment'
data[['Comments', 'Sentiment']] = data['Combined'].str.split(',', n=1, expand=True)

# Drop duplicates and missing values
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# Preprocessing the text
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['Comments'] = data['Comments'].apply(preprocess_text)

# Removing the classes with fewer than 2 samples
data = data.groupby('Sentiment').filter(lambda x: len(x) > 1)

# Spliting the dataset into features and labels
X = data['Comments']
y = data['Sentiment']

# Spliting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#The text data was vectorized using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Training the Logistic Regression model
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=200, class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train_vec, y_train)

model = grid_search.best_estimator_

# Making the predictions
y_pred = model.predict(X_test_vec)
y_pred_proba = model.predict_proba(X_test_vec)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Saving the model and vectorizer
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Saving the predictions to a CSV file
predictions = pd.DataFrame({
    'Comments': X_test,
    'True Sentiment': y_test,
    'Predicted Sentiment': y_pred,
    'Confidence Score': y_pred_proba.max(axis=1)
})
predictions.to_csv('sentiment_predictions.csv', index=False)

print("Model training and evaluation completed. Predictions saved to 'sentiment_predictions.csv'.")

# Definimg Gradio interface
def analyze_sentiment(comment):
    # Normalize input using regex and preprocessing
    comment = preprocess_text(comment)
    comment_vec = vectorizer.transform([comment])
    prediction = model.predict(comment_vec)[0]
    confidence = model.predict_proba(comment_vec).max()
    return f"Sentiment: {prediction} (Confidence: {confidence:.2f})"

interface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter a comment..."),
    outputs=gr.Textbox(label="Predicted Sentiment"),
    title="Sentiment Analysis",
    description="Enter a comment to predict its sentiment with confidence."
)

# Launching Gradio app
interface.launch()