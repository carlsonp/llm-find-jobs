import string
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split


def remove_punctuation(text):
    # keep spaces
    punctuation = string.punctuation.replace(" ", "")
    return text.translate(str.maketrans(punctuation, " " * len(punctuation)))


print("Loading data...")
# Read the file into a list of lines
with open("/data/wikisent2.txt", "r") as file:
    lines = file.readlines()

# Remove newline characters from each line
lines = [line.strip() for line in lines]

# Create a DataFrame from the list of lines
wiki_df = pd.DataFrame(lines, columns=["description"])

wiki_df["job_posting"] = False

job_postings_1_df = pd.read_csv("/data/data job posts.csv")

job_postings_1_df.rename(columns={"jobpost": "description"}, inplace=True)

job_postings_1_df = job_postings_1_df["description"].to_frame()
job_postings_1_df["job_posting"] = True

job_postings_2_df = pd.read_csv("/data/postings.csv")

job_postings_2_df = job_postings_2_df["description"].to_frame()
job_postings_2_df["job_posting"] = True

df = pd.concat([wiki_df, job_postings_1_df, job_postings_2_df], ignore_index=True)

# drop bad records with na values
df = df.dropna()

# remove punctuation
df["description"] = df["description"].apply(remove_punctuation)

print(df.head())
print(df.info())

# Separate features and target
X = df["description"]
y = df["job_posting"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create training and testing DataFrames
train_df = pd.DataFrame({"description": X_train, "job_posting": y_train})
test_df = pd.DataFrame({"description": X_test, "job_posting": y_test})

# Based off example:
# https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html

print("Vectorizing data...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=True,
    sublinear_tf=True,
    max_df=0.5,
    min_df=5,
    stop_words="english",
)
X_train = vectorizer.fit_transform(X_train)

# Extracting features from the test data using the same vectorizer
X_test = vectorizer.transform(X_test)

# feature_names = vectorizer.get_feature_names_out()
# print(feature_names)

# clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
print("Running logistic regression...")
clf = LogisticRegression(C=5, max_iter=1000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)


cm = confusion_matrix(y_test, pred)
print("Confusion Matrix:\n", cm)
print(
    """
   [[true_negative  false_positive]
   [false_negative true_positive]]
"""
)

# Print confusion matrix with percentages without scientific notation
np.set_printoptions(suppress=True, formatter={"float_kind": "{:.4f}".format})
cm = confusion_matrix(y_test, pred, normalize="all")
print("Confusion Matrix (Percentages):\n", cm)

# look for over-fitting, make sure it can generalize
print("Running cross-validation...")
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)
print(f"Cross-validation F1 scores: {scores}")
print(f"Mean F1 score: {scores.mean()}")

print("Serializing data to disk...")
# serialize model to file for future usage
joblib.dump(clf, "/models/job_classifier_model.pkl")
# and the vectorizer
joblib.dump(vectorizer, "/models/vectorizer.pkl")

print("Finished with classification")
