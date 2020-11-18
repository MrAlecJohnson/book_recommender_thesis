import pandas as pd

from numpy import array
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def confidence_hybrid(df, params=None, mappings=None):
    """Adds the probability columns for collaborative filtering and content filtering
    Predicts positive if their sum is at least 1.
    Ignores columns for style-based vectors.
    Ignores params and mappings - they're just for compatibility.
    """
    return (df["cf_probability"] + df["content_probability"]) >= 1


def book_rating_hybrid(df, params=None, mappings=None):
    """Use a book's average rating as a tie-break:
    is it above the average for all books?
    If book is unrated lean towards recommending it
    Ignores columns for style-based vectors.

    mappings must include the key 'average_ratings', which must be an array of ratings
    """
    # Find the average of averages
    ratings = mappings["average_ratings"]
    total_average = sum(ratings.values()) / len(ratings)

    # Map the rows in the results to the indexes of the average_ratings array
    df["book_average"] = df["book_id"].map(ratings)
    df["book_average"].fillna(total_average, inplace=True)

    # Make a prediction for each row based on existing predictions and average rating
    return df.apply(
        lambda row: sum(
            (
                row["cf_class"],
                row["content_class"],
                row["book_average"] >= total_average,
            )
        )
        >= 2,
        axis=1,
    )


def best_of_3_hybrid(df, params=None, mappings=None):
    """Picks best of 3 class from collaborative filtering, vocabulary vectors
    and style vectors. Will raise an exception if no style vectors are included.

    Ignores params and mappings - they're just for compatibility.
    """
    return df.apply(
        lambda row: sum((row["cf_class"], row["content_class"], row["style_class"],))
        >= 2,
        axis=1,
    )


def algo_hybrid(algo, data, params, mappings, training=True):
    """Uses a Scikit Learn algorithm to predict a final class based on predictions
    already made by the collaborative filtering and content-based filtering.

    If training is True, this uses cross-validation to make a final prediction for
    each classification. If training is False, it will fit the hybrid algorithm to
    the whole training set and return the classifier for later use on the test set

    Parameters
    ----------
    algo (class):
        The Scikit Learn classifier to use for hybridising. Designed to work with
        naive bayes, decision trees, logistic regression, SVC and KNN. May work
        with others too

    data (DataFrame):
        The results so far, including columns for true class, predicted classes
        and probabilities

    params (dict):
        Parameters for the hybridisation algorithm

    mappings (dict):
        A previously constructed dictionary of extra data. Must contain
        average rating for each book, number of ratings for each book,
        and number of ratings by each user

    training (bool):
        Whether this experiment is a training run or a test run
    """
    # Preprocess the data and prepare the classifier
    results = pd.Series(dtype=bool)
    df, subset = prepare_hybrid_df(algo, data, params, mappings)
    clf = algo(**params)

    # In training mode, crossvalidate to fit and predict the hybridiser on each fold
    if training:
        sub_kf = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=mappings["random_state"]
        )
        for sub_train, sub_val in sub_kf.split(df, df["true_class"]):
            X_train = df.iloc[sub_train][subset]
            X_val = df.iloc[sub_val][subset]
            y_train = df.iloc[sub_train]["true_class"]
            clf.fit(X_train, y_train)
            # Add each freshly predicted fold to the results series
            results = pd.concat([results, pd.Series(clf.predict(X_val), X_val.index)])

        # Sort it back into index order to undo the crossvalidation shuffling
        results.sort_index(inplace=True)
        return results

    # If a test run, fit to the whole training set and return the classifier
    else:
        clf.fit(df[subset], df["true_class"])
        return clf


def prepare_hybrid_df(algo, df, params, mappings):
    """Adjusts data ready for a hybridisation algorithm to run on it.
    Called from within algo_hybrid.

    Returns an edited DataFrame plus the list of columns the hybridiser will use.
    """

    # For algos other than KNN, add some extra columns to train on
    # These are based on the mappings dictionary
    if algo == KNeighborsClassifier:
        subset = ["cf_probability", "content_probability"]
    else:
        if algo == SVC:
            params["random_state"] = mappings["random_state"]
            params["probability"] = True
        elif algo == DecisionTreeClassifier:
            params["random_state"] = mappings["random_state"]

        user_review_counts = mappings["user_review_counts"]
        book_review_counts = mappings["book_review_counts"]
        ratings_dict = mappings["average_ratings"]
        df["user_review_counts"] = df["user_id"].map(user_review_counts)
        df["book_review_counts"] = df["book_id"].map(book_review_counts)
        df["average_rating"] = df["book_id"].map(ratings_dict)

        df["user_review_counts"].fillna(0, inplace=True)
        df["book_review_counts"].fillna(0, inplace=True)
        df["average_rating"].fillna(df["average_rating"].mean(), inplace=True)
        subset = [
            "cf_probability",
            "content_probability",
            "user_review_counts",
            "book_review_counts",
            "average_rating",
        ]

    # Add the style probability column if present
    if "style_probability" in df:
        subset.append("style_probability")

    # Minmax normalisation needed for some algorithms
    # Not KNN because it only uses probabilities, which are already 0 to 1
    # Not decision tree because each split considers a single feature
    normalise = [MultinomialNB, SVC, LogisticRegression]
    if algo in normalise:
        df[subset] = (df[subset] - df[subset].min()) / (
            df[subset].max() - df[subset].min()
        )
    return df, subset
