import pandas as pd

from copy import copy
from sklearn.neighbors import KNeighborsClassifier


def make_cf_table(full_data, train, classes=3):
    """Produces a set of book vectors based on their user ratings.
    Starts with a matrix of 0s the shape of the full data.
    Adds numbers to the matrix based on the training data.

    Returns dataframe where rows are book goodreads IDs and cols are users

    Parameters
    ----------
    full_data (df):
        All ratings, including the test set, as a dataframe of
        user_id, book_id, rating and recommend

    train (df):
        Training set as a dataframe of user_id, book_id, rating and recommend

    classes (int):
        Set to 2, 3 or 6 to determine how the matrix is filled:
        - if 2, elements are 0 for unread or unejoyed books and 1 for enjoyed books
        - if 3, elements are 0 for unread, 1 for read but disliked, and 2 for enjoyed
        - if 6, elements are 0 for unread and otherwise equal to numeric rating
    """
    empty = pd.DataFrame(
        index=sorted(full_data["book_id"].unique()),
        columns=sorted(full_data["user_id"].unique()),
    )
    if classes == 6:
        known = train.pivot_table(columns="user_id", index="book_id", values="rating")
    else:
        known = train.pivot_table(
            columns="user_id", index="book_id", values="recommend"
        )
        if classes == 2:
            known = known.apply(pd.to_numeric, axis=1)
        elif classes == 3:
            known = known.apply(pd.to_numeric, axis=1) + 1
        else:
            raise ValueError("Please set classes to 2, 3, or 6")
    return empty.combine_first(known).fillna(0)


def splitter(data, train_fold, val_fold, y_col="recommend"):
    """Splits a dataframe into X and y training and validation dataframes

    Returns dataframes for X_train and X_val, and Pandas series for y_train and y_val

    Parameters
    ----------

    data (DataFrame):
        The frame you want to split

    train_fold and val_fold (array-like):
        Index values for the training and validation sets

    y_col (str):
        Name of the column containing your y values - usually "recommend"
    """
    X_train = data.iloc[train_fold].drop(y_col, axis=1)
    X_val = data.iloc[val_fold].drop(y_col, axis=1)
    y_train = data.iloc[train_fold][y_col]
    y_val = data.iloc[val_fold][y_col]
    return X_train, X_val, y_train, y_val


def user_data_cf(X_train, X_val, cf_matrix):
    """Gets rows from a collaborative filtering matrix for the books a user has rated

    Returns two dataframes - one from the training set and one from the validation set

    Parameters
    ----------

    X_train and X_val (array-like):
        Collections of Goodreads book IDs

    cf_matrix (DataFrame):
        The output of make_cf_table() - a matrix of ratings where rows
        are book IDs and columns are users
    """
    book_data_train = cf_matrix.loc[X_train]
    book_data_val = cf_matrix.loc[X_val]
    return book_data_train, book_data_val


def user_data_content(X_train, X_val, vectors, vector_index):
    """Gets specific rows from an array of content vectors, based on Goodreads ID

    Returns two arrays - one from the training set and one from the validation set

    Parameters
    ----------
    X_train and X_val (array-like):
        Collections of Goodreads book IDs

    vectors (array of arrays - eg a sparse matrix):
        Each subarray should be a vector representation of one book
        Make sure the order hasn't changed since creating these vectors

    vector_index (dict):
        Dictionary that maps a Goodread ID onto that book's row number
        in the book vector matrix
    """
    vector_index_train = [vector_index[book] for book in X_train]
    vector_index_val = [vector_index[book] for book in X_val]
    book_data_train = vectors[vector_index_train]
    book_data_val = vectors[vector_index_val]
    return book_data_train, book_data_val


def run_algo(X_train, X_val, y_train, algo, params):
    """Fits a model to the training data, then predicts against validation data

    Called from within predict_user()

    Returns list of tuples in the form (probability false, probability true)
    """
    model = algo(**params)
    model.fit(X_train, y_train)
    return model.predict_proba(X_val)


def predict_user(X_train, X_val, y_train, predict_count, algo, params):
    """Checks and adjusts for edge cases before running the algo on a user's data
    Works on collaborative filtering and content-based vectors.

    Returns list of tuples in the form (probability false, probability true)

    Parameters
    ----------
    X_train, X_val (DataFrame):
        The data you're using for prediction

    y_train (Series):
        The target variable for X_train

    predict_count (int):
        The length of the validation set (eg of X_val)

    algo (class):
        The Scikit Learn class of classification algorithm you want to use for
        prediction. Designed to work with with KNN, SVC, decision tree,
        naive bayes and logistic regression. May work with others too.

    params (dict):
        Dictionary of extra parameters for the prediction algorithm
    """
    train_count = len(y_train)
    positive_count = sum(y_train)

    # If all training data is positive or all negative, skip the algorithm and
    # match this class. Needed as some algorithms require at least 2 classes
    # in training data so won't work in these situations
    if train_count == positive_count:
        return [(0.0, 1.0)] * predict_count
    elif positive_count == 0:
        return [(1.0, 0.0)] * predict_count
    # If you're using KNN with a k larger than the number of ratings in this user's
    # training set, adjust k to match the amount of available data
    elif algo == KNeighborsClassifier and train_count < params["n_neighbors"]:
        adjusted = copy(params)
        if (train_count % 2) == 0:
            train_count -= 1
        adjusted["n_neighbors"] = train_count
        return run_algo(X_train, X_val, y_train, algo, adjusted)
    else:
        return run_algo(X_train, X_val, y_train, algo, params)


def analyse_user(
    user_id,
    X_train,
    X_val,
    y_train,
    y_val,
    cf_matrix,
    vectors,
    vector_index,
    cf_algo,
    cf_params,
    content_algo,
    content_params,
):
    """
    Main function for making collaborative filtering and content-based predictions
    based on a user's training set of book ratings.

    Returns tuples of:
    - index (in original ratings df)
    - user id
    - book id (Goodreads)
    - true class
    - collaborative filtering prediction class
    - content prediction class
    - collaborative filtering prediction probability
    - content prediction probability

    Parameters
    ----------
    user_id (int):
        User's Goodreads ID

    X_train, X_val (DataFrame):
        Predictors for training and validation

    y_train, y_val (array-like of bools):
        Target values for training and validation

    cf_matrix (DataFrame):
        The output of make_cf_table, for looking up collaborative filtering vectors

    vectors (array of arrays):
        The output of whatever vectorisation function the current experiment is using

    vector_index (dict):
        Dictionary mapping Goodreads IDs to rows in the vector array

    cf_algo, content_algo (class):
        The name of the class of algorithm used for predictions. Designed to work with
        with KNN, SVC, decision tree, naive bayes and logistic regression.
        May work with others too.

    cf_params, content_params (dict):
        Dictionary of extra parameters for the prediction algorithms.
    """
    # Construct the subset of data to use for this user
    book_ids_train = X_train["book_id"][X_train["user_id"] == user_id]
    book_ids_val = X_val["book_id"][X_val["user_id"] == user_id]
    y_train_user = y_train[X_train["user_id"] == user_id]
    y_val_user = y_val[X_val["user_id"] == user_id]

    X_train_user_con, X_val_user_con = user_data_content(
        book_ids_train, book_ids_val, vectors, vector_index,
    )
    X_train_user_cf, X_val_user_cf = user_data_cf(
        book_ids_train, book_ids_val, cf_matrix
    )

    # Make predictions for collaborative filtering and content-based filtering
    predict_count = len(y_val_user)
    predict_cf = predict_user(
        X_train_user_cf, X_val_user_cf, y_train_user, predict_count, cf_algo, cf_params
    )
    predict_content = predict_user(
        X_train_user_con,
        X_val_user_con,
        y_train_user,
        predict_count,
        content_algo,
        content_params,
    )

    # Construct a list of results data for this user
    user_id_list = [user_id] * len(y_val_user)
    results = list(
        map(
            list,
            zip(
                y_val_user.index,
                user_id_list,
                book_ids_val,
                y_val_user,
                [True if p[1] >= 0.5 else False for p in predict_cf],
                [True if p[1] >= 0.5 else False for p in predict_content],
                [p[1] for p in predict_cf],
                [p[1] for p in predict_content],
            ),
        )
    )
    return results


def analyse_user_plus_style(
    user_id,
    X_train,
    X_val,
    y_train,
    y_val,
    cf_matrix,
    vectors,
    vector_index,
    cf_algo,
    cf_params,
    content_algo,
    content_params,
):
    """
    Alternative function for making collaborative filtering and content-based
    predictions based on a user's training set of book ratings.

    This version is used when the vectoriser is "separate". This produces 2 sets of
    results for content-based filtering - one based on vocabulary, one on style.

    Needs refactoring to reduce repetition between the 2 functions.

    Returns tuples of:
    - index (in original ratings df)
    - user id
    - book id (Goodreads)
    - true class
    - collaborative filtering prediction class
    - vocabulary prediction class
    - style prediction class
    - collaborative filtering prediction probability
    - content prediction probability
    - style prediction probability

    Parameters
    ----------
    user_id (int):
        User's Goodreads ID

    X_train, X_val (DataFrame):
        Predictors for training and validation

    y_train, y_val (array-like of bools):
        Target values for training and validation

    cf_matrix (DataFrame):
        The output of make_cf_table, for looking up collaborative filtering vectors

    vectors (tuple of arrays):
        The output of the 'separate' vectoriser. The first element of the tuple
        is vocabulary vectors; the second is style vectors

    vector_index (dict):
        Dictionary mapping Goodreads IDs to rows in the vector array

    cf_algo, content_algo (class):
        The name of the class of algorithm used for predictions. Designed to work with
        with KNN, SVC, decision tree, naive bayes and logistic regression.
        May work with others too.

    cf_params, content_params (dict):
        Dictionary of extra parameters for the prediction algorithms.
    """
    # Construct the subset of data to use for this user
    book_ids_train = X_train["book_id"][X_train["user_id"] == user_id]
    book_ids_val = X_val["book_id"][X_val["user_id"] == user_id]
    y_train_user = y_train[X_train["user_id"] == user_id]
    y_val_user = y_val[X_val["user_id"] == user_id]

    X_train_user_con, X_val_user_con = user_data_content(
        book_ids_train, book_ids_val, vectors[0], vector_index,
    )
    X_train_user_style, X_val_user_style = user_data_content(
        book_ids_train, book_ids_val, vectors[1], vector_index,
    )
    X_train_user_cf, X_val_user_cf = user_data_cf(
        book_ids_train, book_ids_val, cf_matrix
    )

    # Make predictions for collaborative filtering and content-based filtering
    predict_count = len(y_val_user)
    predict_cf = predict_user(
        X_train_user_cf, X_val_user_cf, y_train_user, predict_count, cf_algo, cf_params
    )
    predict_content = predict_user(
        X_train_user_con,
        X_val_user_con,
        y_train_user,
        predict_count,
        content_algo,
        content_params,
    )
    predict_style = predict_user(
        X_train_user_style,
        X_val_user_style,
        y_train_user,
        predict_count,
        content_algo,
        content_params,
    )

    # Construct a list of results data for this user
    user_id_list = [user_id] * len(y_val_user)
    results = list(
        map(
            list,
            zip(
                y_val_user.index,
                user_id_list,
                book_ids_val,
                y_val_user,
                [True if p[1] >= 0.5 else False for p in predict_cf],
                [True if p[1] >= 0.5 else False for p in predict_content],
                [True if p[1] >= 0.5 else False for p in predict_style],
                [p[1] for p in predict_cf],
                [p[1] for p in predict_content],
                [p[1] for p in predict_style],
            ),
        )
    )
    return results
