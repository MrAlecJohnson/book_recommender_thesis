import pandas as pd
import numpy as np

from pandas._testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsClassifier

from tests.test_data import varied_ratings, cols3
from modules.analyse import (
    make_cf_table,
    splitter,
    user_data_cf,
    user_data_content,
    run_algo,
    predict_user,
    analyse_user,
)

# TEST DATA
training = varied_ratings.iloc[1:5]

cf_matrix_example = pd.DataFrame(
    [
        [0.0, 2.0, 4.0, 6.0, 8.0],
        [1, 3, 5, 7, 9],
        [0.5, 0.7, 1.1, 3.5, 4.1],
        [100, 140, 480, 621, 89],
        [5.5, 6.5, 7.5, 8.5, 9.5],
    ],
    index=[10, 20, 30, 40, 50],
)
row = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
col = np.array([0, 1, 2, 3, 4, 5, 4, 3, 1, 2])
data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.55,])
book_vectors_example = csr_matrix((data, (row, col)), shape=(5, 6)).toarray()
book_to_row_example = {
    10: 0,
    20: 1,
    30: 2,
    40: 3,
    50: 4,
}

user_X_train_example = pd.DataFrame(
    [
        [1, 0.9, 0.8, 0.7, 0.6],
        [0.6, 0.7, 0.8, 0.9, 1],
        [0.88, 0.71, 0.67, 0.55, 0.91],
        [0.41, 0.45, 0.11, 0.01, 0.39],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
    ]
)
user_X_val_example = pd.DataFrame(
    [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 0.75, 0.81, 0.98, 0.2]],
)
user_y_train_example = pd.Series([True, True, True, False, False, False])
user_y_val_example = pd.Series([True, False, True])


# TESTS
def test_make_cf_table():
    result2 = make_cf_table(varied_ratings, training, classes=2)
    result3 = make_cf_table(varied_ratings, training, classes=3)
    result6 = make_cf_table(varied_ratings, training, classes=6)

    books = [10, 20, 30]
    users = [100, 200, 300]
    matrix2 = pd.DataFrame(
        [[0, 0, 0], [1, 1, 0], [0, 0, 0]], index=books, columns=users
    ).astype(float)
    matrix3 = pd.DataFrame(
        [[0, 1, 1], [2, 2, 0], [0, 0, 0]], index=books, columns=users
    ).astype(float)
    matrix6 = pd.DataFrame(
        [[0, 1, 3], [5, 4, 0], [0, 0, 0]], index=books, columns=users
    ).astype(float)

    assert_frame_equal(result2, matrix2)
    assert_frame_equal(result3, matrix3)
    assert_frame_equal(result6, matrix6)


def test_splitter():
    X_train, X_val, y_train, y_val = splitter(varied_ratings, [0, 1, 2], [3, 4, 5])
    expected_X_train = pd.DataFrame(
        [[100, 10, 2], [100, 20, 5], [200, 10, 1]], columns=cols3, index=[0, 1, 2],
    )
    expected_X_val = pd.DataFrame(
        [[200, 20, 4], [300, 10, 3], [300, 30, 5]], columns=cols3, index=[3, 4, 5],
    )
    expected_y_train = pd.Series(
        [False, True, False], index=[0, 1, 2], name="recommend"
    )
    expected_y_val = pd.Series([True, False, True], index=[3, 4, 5], name="recommend")
    assert_frame_equal(X_train, expected_X_train)
    assert_frame_equal(X_val, expected_X_val)
    assert_series_equal(y_train, expected_y_train)
    assert_series_equal(y_val, expected_y_val)


def test_user_data_cf():
    expected = (
        pd.DataFrame(
            [[0, 2, 4, 6, 8], [0.5, 0.7, 1.1, 3.5, 4.1], [5.5, 6.5, 7.5, 8.5, 9.5],],
            index=[10, 30, 50],
        ),
        pd.DataFrame([[1, 3, 5, 7, 9], [100, 140, 480, 621, 89]], index=[20, 40]),
    )

    result = user_data_cf(
        pd.Series([10, 30, 50]), pd.Series([20, 40]), cf_matrix_example
    )
    assert_frame_equal(
        result[0], expected[0], check_dtype=False,
    )
    assert_frame_equal(
        result[1], expected[1], check_dtype=False,
    )


def test_user_data_content():
    result = user_data_content(
        pd.Series([10, 30, 50]),
        pd.Series([20, 40]),
        book_vectors_example,
        book_to_row_example,
    )
    print(result)
    expected = (
        [[0.1, 0, 0, 0, 0, 0.6], [0, 0, 0.3, 0.8, 0, 0], [0, 0, 0.55, 0, 0.5, 0]],
        [[0, 0.2, 0, 0, 0.7, 0], [0, 0.9, 0, 0.4, 0, 0]],
    )
    assert_array_equal(
        result[0], expected[0],
    )
    assert_array_equal(
        result[1], expected[1],
    )


def test_run_algo():
    result = run_algo(
        user_X_train_example,
        user_X_val_example,
        user_y_train_example,
        KNeighborsClassifier,
        {},
    )
    assert len(result) == len(user_X_val_example)
    for r in result:
        assert len(r) == 2
        assert 0 <= r[0] <= 1
        assert 0 <= r[1] <= 1


def test_predict_user():
    all_positive = predict_user(
        user_X_train_example.iloc[0:3],
        user_X_val_example,
        user_y_train_example.iloc[0:3],
        3,
        KNeighborsClassifier,
        {},
    )
    assert all_positive == [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    all_negative = predict_user(
        user_X_train_example.iloc[3:],
        user_X_val_example,
        user_y_train_example.iloc[3:],
        3,
        KNeighborsClassifier,
        {},
    )
    assert all_negative == [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]

    not_enough_training = predict_user(
        user_X_train_example,
        user_X_val_example,
        user_y_train_example,
        3,
        KNeighborsClassifier,
        {"n_neighbors": 7},
    )
    # Shows KNN has adjusted neighbour count to 5
    possible = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for result in not_enough_training:
        assert result[0] in possible
        assert result[1] in possible


def test_analyse_user():
    cols2 = ["user_id", "book_id"]
    X_train = pd.DataFrame(
        [[100, 10], [100, 20], [100, 30], [200, 10], [0, 20],], columns=cols2
    )
    X_val = pd.DataFrame([[100, 40], [100, 50], [1000, 30], [10, 20],], columns=cols2)
    y_train = pd.Series([False, True, True, False, False])
    y_val = pd.Series([True, False, False, False])

    result = analyse_user(
        100,
        X_train,
        X_val,
        y_train,
        y_val,
        cf_matrix_example,
        book_vectors_example,
        book_to_row_example,
        KNeighborsClassifier,
        {"n_neighbors": 3},
        KNeighborsClassifier,
        {"n_neighbors": 3},
    )
    assert result[0][:-2] == [0, 100, 40, True, True, True]
    assert result[0][-1] - 0.666667 <= 0.0001
    assert result[0][-2] - 0.666667 <= 0.0001
    assert result[1][:-2] == [1, 100, 50, False, True, True]
    assert result[1][-1] - 0.666667 <= 0.0001
    assert result[1][-2] - 0.666667 <= 0.0001
