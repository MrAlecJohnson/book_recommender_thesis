import pandas as pd

from modules.evaluate import (
    stats,
    count_true,
    evaluate_results,
    average_folds,
)


df = pd.DataFrame(
    [
        [True, True, False, True],
        [False, False, True, False],
        [True, False, False, True],
        [False, True, True, False],
    ],
    columns=["true_class", "cf_class", "content_class", "hybrid_class"],
)


settings = [
    "A short description",
    10,
    10,
    "vectoriser",
    str({"vectoriser_params": 0}),
    "algo1",
    str({"params1": 10}),
    "algo2",
    str({"params2": 20}),
    "algo3",
    str({"params3": 30}),
]

expected = [
    0.5,
    0.5,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    0.5,
    0.5,
    0,
    0,
    2,
    2,
    0,
    0,
    2,
    2,
    0,
    0,
    1,
    1,
]


def test_stats():
    tp = 10
    tn = 20
    fp = 5
    fn = 25

    assert stats(tp, tn, fp, fn) == [0.6667, 0.2857, 0.40, 0.50]


def test_count_true():
    assert count_true(df, "cf_class") == (1, 1, 1, 1,)
    assert count_true(df, "content_class") == (0, 0, 2, 2,)
    assert count_true(df, "hybrid_class") == (2, 2, 0, 0,)


def test_evaluate_results():
    assert evaluate_results(df) == expected


def test_average_folds():
    processed = [
        expected,
        [
            1,
            1,
            0.5,
            0.5,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            10,
            20,
            0,
            6,
            0,
            0,
            4,
            4,
            11,
            11,
            0.67,
            1,
        ],
    ]
    assert average_folds(processed) == [
        0.75,
        0.75,
        0.25,
        0.25,
        0.5,
        0.5,
        1,
        1,
        1,
        0.5,
        0.75,
        0.25,
        5,
        10,
        1,
        4,
        0,
        0,
        3,
        3,
        5.5,
        5.5,
        0.835,
        1,
    ]
