import os
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


'''
Vectorize the text using Word Count
- Input are dataframes with a column 'exec_trace' that contains the text to be vectorized
'''
def feat_extract_wc(X_train, X_test):
    # Combine the list of traces into a single trace
    X_train["exec_trace_combined"] = X_train["exec_trace"].apply(lambda a: " ".join(a) if isinstance(a, list) else "")
    X_test["exec_trace_combined"] = X_test["exec_trace"].apply(lambda a: " ".join(a) if isinstance(a, list) else "")

    wc_vectorizer = CountVectorizer(
        min_df=3,
        max_features=600,
    )

    X_train_wc = wc_vectorizer.fit_transform(X_train["exec_trace_combined"])
    X_test_wc = wc_vectorizer.transform(X_test["exec_trace_combined"])

    return X_train_wc, X_test_wc


'''
Vectorize the text using TF-IDF
- Input are dataframes with a column 'exec_trace' that contains the text to be vectorized
'''
def feat_extract_tfidf(X_train, X_test):
    # Combine the list of traces into a single trace
    X_train["exec_trace_combined"] = X_train["exec_trace"].apply(lambda a: " ".join(a) if isinstance(a, list) else "")
    X_test["exec_trace_combined"] = X_test["exec_trace"].apply(lambda a: " ".join(a) if isinstance(a, list) else "")

    tfidf_vectorizer = TfidfVectorizer(
        min_df=3,
        max_features=600,
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train["exec_trace_combined"])
    X_test_tfidf = tfidf_vectorizer.transform(X_test["exec_trace_combined"])
    return X_train_tfidf, X_test_tfidf


'''
Vectorize the text using boolean matrix
- Input are dataframes with a column 'exec_trace' that contains the text to be vectorized
'''
def feat_extract_boolean_matrix(X_train, X_test):
    # Combine the list of traces into a single trace
    X_train["exec_trace_combined"] = X_train["exec_trace"].apply(lambda a: " ".join(a) if isinstance(a, list) else "")
    X_test["exec_trace_combined"] = X_test["exec_trace"].apply(lambda a: " ".join(a) if isinstance(a, list) else "")

    # Extract unique strings from training data
    unique_strings = set()

    for trace in X_train["exec_trace_combined"]:
        unique_strings.update(trace.split())

    # Convert to a list
    unique_strings = list(unique_strings)

     # Create the boolean matrix for the training dataset
    train_matrix = []
    for trace in X_train["exec_trace_combined"]:
        trace_words = set(trace.split())
        row = [string in trace_words for string in unique_strings]
        train_matrix.append(row)

    train_boolean_df = pd.DataFrame(train_matrix, columns=unique_strings)

    # Create the boolean matrix for the test dataset
    test_matrix = []
    for trace in X_test["exec_trace_combined"]:
        trace_words = set(trace.split())
        row = [string in trace_words for string in unique_strings]
        test_matrix.append(row)

    test_boolean_df = pd.DataFrame(test_matrix, columns=unique_strings)

    print(unique_strings)

    return train_boolean_df, test_boolean_df