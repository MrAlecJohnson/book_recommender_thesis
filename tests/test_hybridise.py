import pandas as pd

from numpy.testing import assert_array_equal
from sklearn.tree import DecisionTreeClassifier
from modules.hybridise import (
    confidence_hybrid,
    book_rating_hybrid,
    algo_hybrid,
)

columns = [
    "cf_probability",
    "content_probability",
    "cf_class",
    "content_class",
    "book_id",
    "user_id",
    "true_class",
]

df = pd.DataFrame(
    [
        [0, 1, False, True, 1, 100, True],
        [0.4, 0.8, False, True, 2, 100, False],
        [0.7, 0.1, True, False, 3, 100, True],
        [0.6, 0.45, True, False, 1, 200, False],
        [0.2, 0.6, False, True, 2, 200, False],
        [0.3, 0.4, False, False, 3, 300, True],
        [0.99, 0.71, True, True, 2, 300, True],
    ],
    columns=columns,
)

df2 = pd.DataFrame(
    [
        [1, 1, True, True, 1, 100, True],
        [0, 0, False, False, 2, 100, False],
        [0.9, 0.9, True, True, 3, 100, True],
        [0.1, 0.1, False, False, 1, 200, False],
        [0.6, 0.01, True, False, 10, 200, False],
        [0.9, 0.9, True, True, 1, 100, True],
        [0.1, 0.1, False, False, 2, 100, False],
        [0.9, 0.9, True, True, 3, 100, True],
        [0.1, 0.1, False, False, 1, 200, False],
        [0.49, 0.995, False, True, 11, 200, True],
        [0.4, 0.91, False, True, 11, 200, True],
        [0.55, 0.12, True, False, 10, 200, False],
    ],
    columns=columns,
)


def test_confidence_hybrid():
    assert_array_equal(
        confidence_hybrid(df), [True, True, False, True, False, False, True]
    )


ratings_dict_1 = {
    1: 4,
    2: 3,
    3: 2,
}

ratings_dict_2 = {
    1: 4.89,
    2: 1.23,
    3: 3.56,
}


def test_book_rating_hybrid():
    assert_array_equal(
        book_rating_hybrid(df, mappings={"average_ratings": ratings_dict_2}),
        [True, False, True, True, False, False, True],
    )


def test_algo_hybrid():
    results = algo_hybrid(
        DecisionTreeClassifier,
        df2,
        {"criterion": "gini", "max_depth": 10},
        {
            "average_ratings": {1: 3, 2: 3, 3: 3, 10: 1, 11: 5},
            "book_review_counts": {1: 20, 2: 20, 3: 20, 10: 1, 11: 100},
            "user_review_counts": {100: 20, 200: 20, 300: 20},
            "random_state": 50,
        },
    )
    assert_array_equal(
        results,
        [True, False, True, False, False, True, False, True, False, True, True, False],
    )
