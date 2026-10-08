import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# NEWS TEXT CLASSIFICATION SYSTEM
# -------------------------------------------------

# Sample news dataset
data = {
    "text": [
        "India wins the cricket match against Australia",
        "Virat Kohli scores a brilliant century in the match",
        "The football team wins the championship",
        "Olympic athletes prepare for the upcoming games",
        "Government announces new education policy",
        "Parliament passes a new bill today",
        "Prime Minister addresses the nation",
        "Election campaign begins across the country",
        "Apple launches a new smartphone with advanced features",
        "Microsoft introduces a new artificial intelligence tool",
        "Technology companies invest heavily in artificial intelligence",
        "New computer processor improves performance",
        "Stock market rises after strong economic growth",
        "Company reports record quarterly profit",
        "Banks announce changes in interest rates",
        "Investors are optimistic about the financial market"
    ],

    "category": [
        "Sports",
        "Sports",
        "Sports",
        "Sports",
        "Politics",
        "Politics",
        "Politics",
        "Politics",
        "Technology",
        "Technology",
        "Technology",
        "Technology",
        "Business",
        "Business",
        "Business",
        "Business"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

print("=" * 60)
print("        NEWS TEXT CLASSIFICATION SYSTEM")
print("=" * 60)

print("\nDataset:")
print(df.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["category"],
    test_size=0.25,
    random_state=42,
    stratify=df["category"]
)

# Create NLP pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nAvailable Categories:")
print("1. Sports")
print("2. Politics")
print("3. Technology")
print("4. Business")

# -------------------------------------------------
# USER NEWS CLASSIFICATION
# -------------------------------------------------

while True:

    print("\n" + "-" * 60)
    news = input("Enter a news headline (or type 'exit' to stop): ")

    if news.lower() == "exit":
        print("\nThank you for using the News Text Classification System!")
        break

    if news.strip() == "":
        print("Please enter some news text.")
        continue

    # Predict category
    prediction = model.predict([news])[0]

    # Prediction probabilities
    probabilities = model.predict_proba([news])[0]
    confidence = max(probabilities) * 100

    print("\nNews:", news)
    print("Predicted Category:", prediction)
    print("Confidence:", round(confidence, 2), "%")