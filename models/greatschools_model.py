#!/usr/bin/env python3

"""
"""

import sys

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np
from scipy.sparse import hstack, vstack
import pandas as pd

import lda_topics
from greatschools_crawler import GreatSchoolsCrawler

NUM_LDA_TOPICS = 10

FIELDS_TO_TARGET = ["progress_rating", "test_score_rating", "overall_rating", "equity_rating"]
    
def generate_feature_vectors(crawler, lda):
    for x in crawler.generate_snippets():
        topic_vector = lda_topics.snippet_to_lda_vector(x, lda)
        tokens = lda_topics.spacy_tokenizer(x["content"]).split()

        # Note:  I'm reusing the vectorizer used to fit the LDA model to do the
        # vectorizing for the final model.  There's no particular reason other than
        # convenience for this; a vectorizer with difference cutoffs might do better.
        review_words_vectorized = lda["vectorizer"].transform([" ".join(tokens)])

        # Also include the topic vector itself
        additional_features = pd.Series(topic_vector)
        features = hstack([review_words_vectorized,
                           additional_features])

        yield {"snippet": x, "features": features}

def train_model():
    crawler = GreatSchoolsCrawler()

    # Generate LDA model
    print("Training LDA model...")
    lda = lda_topics.train_lda_model(crawler, num_topics=NUM_LDA_TOPICS)
    lda_topics.print_selected_topics(lda)

    vocab = lda["vectorizer"].get_feature_names()
    vocab_size = len(vocab)
    topic_vector_size = NUM_LDA_TOPICS
    print("Vocab size: %d, num topics: %d" % (vocab_size, topic_vector_size))
    feature_names = (["review_has_%s" % (vocab[d]) for d in range(vocab_size)] +
                     ["topic_vector_%d" % (d) for d in range(topic_vector_size)])

    all_targets = {x: [] for x in FIELDS_TO_TARGET}
    all_features = []  # universal feature set
    for res in generate_feature_vectors(crawler, lda):
        all_features.append(res["features"])

        for x in FIELDS_TO_TARGET:
            decision = (float(res["snippet"][x]) >= 8.0 and 1.0 or 0.0)
            all_targets[x].append(decision)

    features_matrix = vstack(all_features)
    print("Training feature matrix shape:", features_matrix.shape)

    classifiers = {}  # task -> classifier
    for task in FIELDS_TO_TARGET:
        print("Training model for %s prediction..." % (task))
        classifiers[task] = LogisticRegression(solver='sag', max_iter=100)
        targets = all_targets[task]
        auc = np.mean(cross_val_score(classifiers[task], features_matrix,
                                      np.array(targets), cv=5, n_jobs=-1,
                                      scoring='roc_auc'))
        print("AUC (for %s prediction): %f" % (task, auc))

        # Train final model
        classifiers[task].fit(features_matrix, np.array(targets))

        # Print feature importances
        importances = []
        # Requires converting to a CSR-format sparse matrix first
        features_matrix_csr = features_matrix.tocsr()
        for i, weight in enumerate(classifiers[task].coef_[0]):
            column = features_matrix_csr[:, i].toarray()
            stddev = np.std(column)
            importances.append((feature_names[i], weight * stddev))
        importances.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, weight in importances[:30]:
            print("   ", name, round(weight, 2))


if __name__ == '__main__':
    train_model()
