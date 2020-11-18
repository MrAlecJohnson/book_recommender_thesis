import sqlite3
import pandas as pd

from pathlib import Path

DATA_DIR = Path.cwd() / "data"


def book_data(
    filename="book_data_cut.csv", location=DATA_DIR,
):
    """Opens the book metadata as a dataframe - defaults to my local file structure
    """
    return pd.read_csv(location / filename)


def map_book_ids(direction, location=DATA_DIR, filename="book_data_cut.csv"):
    """direction = gute_to_gr or gr_to_gute
    """
    books = book_data(filename, location)
    goodreads_ids = books["Goodreads ID"]
    gutenberg_ids = books["Catalogue number"]
    if direction == "gute_to_gr":
        return {gute: gr for gr, gute in zip(goodreads_ids, gutenberg_ids)}
    elif direction == "gr_to_gute":
        return {gr: gute for gr, gute in zip(goodreads_ids, gutenberg_ids)}
    else:
        raise ValueError(
            "Please specify a direction for the dictionary.\n"
            "Use either 'gute_to_gr' or 'gr_to_gute'"
        )


def ratings_data(
    filename="book_ratings.db", location=DATA_DIR,
):
    """Opens the sqlite database of book ratings and returns it as a Pandas dataframe
    """
    conn = sqlite3.connect(location / filename)
    df = pd.DataFrame(
        conn.execute("SELECT * FROM book_ratings"),
        columns=["book_id", "user_id", "rating"],
    )
    conn.close()

    return df


def trim_ratings(ratings, user_min=20, book_min=10):
    """Takes a ratings dataframe and drops all rows where:
    - the user_id has given fewer than user_min ratings
    - the book_id has been given fewer than book_min ratings
    """
    user_counts = ratings["user_id"].value_counts()
    enough_users = user_counts[user_counts >= user_min]

    book_counts = ratings["book_id"].value_counts()
    enough_books = book_counts[book_counts >= book_min]

    return ratings[
        (ratings["user_id"].isin(enough_users.index))
        & (ratings["book_id"].isin(enough_books.index))
    ]


def set_threshold(df, threshold):
    """Takes a ratings dataframe and returns a series of booleans specifying
    whether the rating given reached a given threshold
    """
    return df["rating"] >= threshold


def set_class_proportions(df, low=0.01, high=0.99):
    """Sets limits on the proportion of a user's ratings that are positive.
    Default values remove users who always rate positive or always rate negative.
    Inclusive, so set to 0 and 1 to include everyone
    """
    # Pivot to a table of users and their number of True and False ratings
    user_rating_counts = df.pivot_table(
        index="user_id", columns="recommend", aggfunc="size", fill_value=0
    )
    # Find the proportion of ratings that are positive for each user
    user_rating_counts["proportion"] = user_rating_counts[True] / (
        user_rating_counts[True] + user_rating_counts[False]
    )
    # Drop users whose positive proportion isn't between the low and high parameters
    user_rating_counts["pass"] = user_rating_counts["proportion"].between(
        low, high, inclusive=True
    )
    return df[df["user_id"].map(user_rating_counts["pass"])]


def read_books(ids, location=DATA_DIR, folder="gutenberg_processed"):
    """Reads a list of Gutenberg book IDs and returns
    a dictionary of the book content, indexed by Goodreads ID
    """
    texts = {}
    gutenberg_to_gr = map_book_ids("gute_to_gr", location)
    for i in ids:
        filename = location / folder / f"{i}_processed.txt"
        with open(filename, "r") as f:
            gr = gutenberg_to_gr[i]
            texts[gr] = f.read()
    return texts
