#!/usr/bin/env python3

"""
Creates an LDA topic model on the LVN training corpus
"""

import json
import pandas as pd
import string

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# Number of document clusters to produce if num_topics isn't set
DEFAULT_NUM_TOPICS = 10

MIN_DF = 5

# See https://www.kaggle.com/thebrownviking20/topic-modelling-with-spacy-and-scikit-learn
parser = English()
stopwords = list(STOP_WORDS)
punctuations = string.punctuation

def spacy_tokenizer(sentence, use_stopwords=False):
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-"
                else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if ((not use_stopwords or word not in stopwords)
                                              and word not in punctuations)]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

# Functions for printing keywords for each topic
def print_selected_topics(lda, top_n=10):
    for idx, topic in enumerate(lda["lda_model"].components_):
        print("Topic %d:" % (idx))
        print([lda["vectorizer"].get_feature_names()[i]
               for i in topic.argsort()[:-top_n - 1:-1]])

def train_lda_model(crawler, num_topics=DEFAULT_NUM_TOPICS):
    num_snippets = 0
    num_words = 0
    num_speaker_turns = 0
    data = {"docs": []}

    last_speaker = None
    for x in crawler.generate_snippets():
        data["docs"].append(spacy_tokenizer(x["content"]))
        if "speaker_id" in x:
            if (x["conversation_id"], x["speaker_id"]) != last_speaker:
                num_speaker_turns += 1
            num_snippets += 1
            num_words += len(x["words"])
            last_speaker = (x["conversation_id"], x["speaker_id"])
    df = pd.DataFrame(data, columns=["docs"])

    print("Processing %d snippets with %d words and %d speaker turns" % (
        num_snippets, num_words, num_speaker_turns))
    vectorizer = CountVectorizer(min_df=MIN_DF, max_df=0.9, stop_words='english', lowercase=True,
                                 token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    data_vectorized = vectorizer.fit_transform(df["docs"])
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10,
                                    learning_method='online', verbose=True)
    data_lda = lda.fit_transform(data_vectorized)   # use output?
    return {"lda_model": lda, "vectorizer": vectorizer}

def snippet_to_lda_vector(x, lda):
    doc = spacy_tokenizer(x["content"])
    topic_vector = lda["lda_model"].transform(lda["vectorizer"].transform([doc]))[0]
    return list(topic_vector)

