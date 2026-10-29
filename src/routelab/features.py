"""Persisted prompt-only TF-IDF and structural transforms; no execution outcomes enter features."""

import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def structural(texts) -> np.ndarray:
    rows = []
    for text in texts:
        rows.append(
            [
                len(text.split()),
                len(text),
                text.count("\n") + 1,
                text.count("?"),
                len(re.findall(r"\d", text)),
                text.count("```"),
                text.count("{"),
                text.count(":"),
                text.count(";"),
                text.count("("),
            ]
        )
    return np.log1p(np.asarray(rows, dtype=float))


class PromptFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, max_features: int = 6000):
        self.max_features = max_features

    def fit(self, texts, y=None):
        texts = list(texts)
        self.tfidf_ = TfidfVectorizer(
            max_features=self.max_features, ngram_range=(1, 2), sublinear_tf=True
        )
        self.tfidf_.fit(texts)
        self.scaler_ = StandardScaler().fit(structural(texts))
        return self

    def transform(self, texts):
        texts = list(texts)
        return hstack(
            [self.tfidf_.transform(texts), csr_matrix(self.scaler_.transform(structural(texts)))],
            format="csr",
        )
