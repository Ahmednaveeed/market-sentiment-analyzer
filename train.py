import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from vectorizer import create_vectorizer
from preprocess import preprocess_text

# Load and clean dataset
df = pd.read_csv('synthetic_tweets.csv')
df.dropna(inplace=True)

# Preprocess tweets
df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)

# Encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

# Features and target
X = df['cleaned_tweet']
y = df['label_encoded']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize training data
x_train_vectorized, vectorizer = create_vectorizer(x_train)

# Train model
model = LogisticRegression()
model.fit(x_train_vectorized, y_train)

# Evaluate model
x_test_vectorized = vectorizer.transform(x_test)
y_pred = model.predict(x_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model, vectorizer, and label encoder saved.")
