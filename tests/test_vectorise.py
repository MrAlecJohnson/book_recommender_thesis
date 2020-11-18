import numpy as np

from scipy.sparse import csr_matrix
from numpy.testing import assert_array_equal, assert_allclose
from modules.vectorise import (
    vectorise_count,
    vectorise_tfidf,
    vectorise_lsi,
)

text1 = "tigers tigers tigers consume animals unless animals escape"
text2 = "giraffes consume tasty leaves leaves"
text3 = "tigers escape unless unless tasty"
text4 = "giraffes tigers also baboons"
texts = {1: text1, 2: text2, 3: text3, 4: text4}


def test_vectorise_count():
    row = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3])
    col = np.array([7, 6, 2, 1, 0, 5, 4, 3, 1, 7, 6, 5, 2, 6, 3])
    data = np.array([1, 3, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1])
    expected = csr_matrix((data, (row, col)), shape=(4, 8))

    result = vectorise_count(
        texts, [1, 2, 3], {"analyzer": "word", "ngram_range": [1, 1]}
    )
    assert_array_equal(result.toarray(), expected.toarray())


def test_vectorise_tfidf():
    row = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3])
    col = np.array([7, 6, 2, 1, 0, 5, 4, 3, 1, 7, 6, 5, 2, 6, 3])
    data = np.array(
        [
            0.22992676437941745,
            0.6897802931382524,
            0.22992676437941745,
            0.22992676437941745,
            0.6046521283053111,
            0.3065042162415877,
            0.8060324216071015,
            0.40301621080355077,
            0.3065042162415877,
            0.7559289460184545,
            0.37796447300922725,
            0.37796447300922725,
            0.37796447300922725,
            0.6053485081062916,
            0.7959605415681652,
        ]
    )
    expected = csr_matrix((data, (row, col)), shape=(4, 8))

    result = vectorise_tfidf(
        texts, [1, 2, 3], {"analyzer": "word", "ngram_range": [1, 1]}
    )
    print(result)
    assert_allclose(result.toarray(), expected.toarray())


def test_vectorise_lsi():
    expected = np.array(
        [
            [0.969207, -0.123008, -0.087589],
            [0.301727, 0.901568, -0.309197],
            [0.772099, 0.237163, 0.585673],
            [0.844425, -0.39781, -0.324496],
        ]
    )
    result = vectorise_lsi(
        texts,
        [1, 2, 3],
        {
            "tfidf": {
                "analyzer": "word",
                "ngram_range": [1, 1],
                "min_df": 2,
                "lowercase": False,
            },
            "lsi": {"n_components": 3, "algorithm": "randomized", "random_state": 50},
            "normalise": False,
        },
    )
    assert_allclose(result, expected, rtol=1e-05)
