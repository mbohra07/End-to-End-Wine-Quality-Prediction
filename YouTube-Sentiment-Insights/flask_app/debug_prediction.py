import mlflow
import pickle
from mlflow.tracking import MlflowClient
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_comment(comment):
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return comment

# Load model and vectorizer
mlflow.set_tracking_uri("http://ec2-13-202-137-74.ap-south-1.compute.amazonaws.com:5000/")
model = mlflow.pyfunc.load_model("models:/my_model/2")

with open("/Users/madhurambohra/Desktop/Ultimate-MLOps-Full-Course/YouTube-Sentiment-Insights/tfidf_vectorizer.pkl", 'rb') as file:
    vectorizer = pickle.load(file)

# Test prediction
comments = ['This video is awesome: way too good, i mean crazyyy', 'Very bad explanation, poor video']
print("Original comments:", comments)

preprocessed = [preprocess_comment(comment) for comment in comments]
print("Preprocessed:", preprocessed)

transformed = vectorizer.transform(preprocessed)
print("Transformed shape:", transformed.shape)

dense = transformed.toarray()
print("Dense shape:", dense.shape)

predictions = model.predict(dense)
print("Predictions:", predictions)
print("Prediction type:", type(predictions))