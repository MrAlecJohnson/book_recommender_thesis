import pandas as pd

from pathlib import Path
from numpy.testing import assert_array_equal, assert_raises
from pandas._testing import assert_frame_equal
from modules.load_data import (
    set_class_proportions,
    set_threshold,
    trim_ratings,
    read_books,
)
from tests.test_data import (
    cols3,
    threshold_test,
    varied_review_counts,
)


def test_trim_ratings():
    keep_1 = trim_ratings(varied_review_counts, 1, 1)
    keep_2_user = trim_ratings(varied_review_counts, 2, 1)
    keep_2_book = trim_ratings(varied_review_counts, 1, 2)
    keep_3_user = trim_ratings(varied_review_counts, 3, 1)
    keep_3_book = trim_ratings(varied_review_counts, 1, 3)
    too_many_user = trim_ratings(varied_review_counts, 100, 1)
    too_many_book = trim_ratings(varied_review_counts, 1, 100)

    assert_frame_equal(keep_1, varied_review_counts)
    assert_array_equal(
        keep_2_user, [[1, 20, 1], [1, 20, 1], [2, 30, 5], [2, 30, 5], [2, 30, 4]]
    )
    assert_array_equal(
        keep_2_book, [[1, 20, 1], [1, 20, 1], [2, 30, 5], [2, 30, 5], [2, 30, 4]]
    )
    assert_array_equal(keep_3_user, [[2, 30, 5], [2, 30, 5], [2, 30, 4]])
    assert_array_equal(keep_3_book, [[2, 30, 5], [2, 30, 5], [2, 30, 4]])
    assert_array_equal(too_many_user, pd.DataFrame([], columns=cols3))
    assert_array_equal(too_many_book, pd.DataFrame([], columns=cols3))


def test_set_threshold():
    assert_array_equal(
        set_threshold(threshold_test, 1), [True, True, True, True, True],
    )
    assert_array_equal(
        set_threshold(threshold_test, 2), [False, True, True, True, True],
    )
    assert_array_equal(
        set_threshold(threshold_test, 3), [False, False, True, True, True],
    )
    assert_array_equal(
        set_threshold(threshold_test, 4), [False, False, False, True, True],
    )
    assert_array_equal(
        set_threshold(threshold_test, 5), [False, False, False, False, True],
    )


def test_set_class_proportions():
    df = pd.DataFrame(
        [
            [1, True],
            [1, False],
            [2, True],
            [3, False],
            [4, True],
            [4, False],
            [4, False],
            [4, False],
            [5, False],
            [5, True],
            [5, True],
            [5, True],
        ],
        columns=["user_id", "recommend"],
    )

    drop_all_true = [
        [1, True],
        [1, False],
        [3, False],
        [4, True],
        [4, False],
        [4, False],
        [4, False],
        [5, False],
        [5, True],
        [5, True],
        [5, True],
    ]
    drop_all_false = [
        [1, True],
        [1, False],
        [2, True],
        [4, True],
        [4, False],
        [4, False],
        [4, False],
        [5, False],
        [5, True],
        [5, True],
        [5, True],
    ]
    drop_most_true = [
        [1, True],
        [1, False],
        [3, False],
        [4, True],
        [4, False],
        [4, False],
        [4, False],
    ]
    drop_most_false = [
        [1, True],
        [1, False],
        [2, True],
        [5, False],
        [5, True],
        [5, True],
        [5, True],
    ]
    drop_all_but_equal = [[1, True], [1, False]]

    assert_array_equal(set_class_proportions(df, 0, 1), df)
    assert_array_equal(set_class_proportions(df, 0, 0.9), drop_all_true)
    assert_array_equal(set_class_proportions(df, 0.1, 1), drop_all_false)
    assert_array_equal(set_class_proportions(df, 0, 0.6), drop_most_true)
    assert_array_equal(set_class_proportions(df, 0.4, 1), drop_most_false)
    assert_array_equal(set_class_proportions(df, 0.5, 0.5), drop_all_but_equal)

    assert_raises(
        AssertionError, assert_array_equal, set_class_proportions(df, 0.5, 0.5), df
    )


def test_read_books():
    assert read_books([9993, 500], Path.cwd() / "tests/test_files", "") == {
        6059070: "first book test alec potato potato\n",
        14327844: "alec test book immediately second second\n",
    }
