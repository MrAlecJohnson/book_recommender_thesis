import pickle
import random
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import NMF

DATA_DIR = Path.cwd() / "data"
MODEL_DIR = Path.cwd() / "models"


def min_max(arrays):
    """Helper function. Takes array of arrays and scales each column to go from 0 to 1
    Scaling is based on minimum and maximum of each column.
    Returns array of arrays.
    """
    df = pd.DataFrame(np.row_stack(arrays))
    df = (df - df.min()) / (df.max() - df.min())
    return df.to_numpy()


def vectorise_count(texts, training_ids, params):
    """Counts occurrences of each word in each text.
    No longer used - prefer vectorise_tfidf below
    """
    train = [texts[i] for i in training_ids]
    vectoriser = CountVectorizer(input="content", **params)
    vectoriser.fit(train)
    return vectoriser.transform(texts.values())


def vectorise_tfidf(texts, training_ids, params):
    """Calculates tfidf weights. Trains on the training set alone,
    then transforms all the books in the training AND test set.
    Returns Scikit Learn sparse matrix of tfidf values for all texts

    Parameters
    ----------
    texts (dict):
        Dictionary where keys are Goodreads IDs and values are full processed texts
        The books should include everything in the training and test data

    training_ids (list):
        The Goodreads IDs of the books in the training set

    params (dict):
        Parameters for the Scikit Learn TfidfVectoriser
    """
    train = [texts[i] for i in training_ids]
    vectoriser = TfidfVectorizer(input="content", **params)
    vectoriser.fit(train)
    return vectoriser.transform(texts.values())


def vectorise_lsi(texts, training_ids, params):
    """First gets count or tfidf values, then performs LSI to reduce dimensionality
    Returns a dense matrix of LSI values.

    Parameters
    ----------
    texts (dict):
        Dictionary where keys are Goodreads IDs and values are full processed texts
        The books should include everything in the training and test data

    training_ids (list):
        The Goodreads IDs of the books in the training set

    params (dict):
        Dictionary with 3 keys:
        - "tfidf" or "count", a subdict containing parameters for initial vectoriser
        - "lsi", a subdict containing params for Scikit Learn's TruncatedSVD class
        - "normalise", a boolean specifying whether to min/max normalise the data
            between the vectorising and the LSI
    """
    if "tfidf" in params:
        vectors = vectorise_tfidf(texts, training_ids, params["tfidf"])
    elif "count" in params:
        vectors = vectorise_count(texts, training_ids, params["count"])
    lsi = TruncatedSVD(**params["lsi"])
    result = lsi.fit_transform(vectors)
    if params["normalise"]:
        return min_max(result)
    else:
        return result


def vectorise_nmf(texts, training_ids, params):
    """NOT USED. Should perform nonnegative matrix factorisation on tfidf or count values
    However, too slow for use without further experimentation (eg reducing vocab size)
    """
    if "tfidf" in params:
        vectors = vectorise_tfidf(texts, training_ids, params["tfidf"])
    elif "count" in params:
        vectors = vectorise_count(texts, training_ids, params["count"])
    nmf = NMF(**params["nmf"])
    result = nmf.fit_transform(vectors)
    normalised = Normalizer(copy=False).fit_transform(result)
    return normalised


def use_styles(texts, training_ids, params):
    """Returns an array of style vectors in the order of texts.keys()

    Loads the style vectors specified in params["location"], then maps
    texts.keys() onto these new vectors

    If params["normalise"], scale each feature from 0 to 1
    """
    style_vectors = pickle.load(open(Path.cwd() / params["location"], "rb"))
    data = [style_vectors[t] for t in texts.keys()]
    if params["normalise"]:
        return min_max(data)
    else:
        return np.array(data)


def random_vectors(texts, training_ids, params):
    """Produces random vectors for the texts, to check my recommender actually helps

    params should contain a random seed and a length for the random vectors
    """
    random.seed(params["random_state"])
    data = [[random.random() for r in range(params["length"])] for t in texts.keys()]
    return np.array(data)


def style_lsi_combined(texts, training_ids, params):
    """Joins LSI vector and style vector into a single vector for each text

    params should be a dict of two keys:
        - "vocab", containing the params you'd put into vectorise_lsi()
        - "style", containing the params you'd to into use_styles()
    """
    vocab = vectorise_lsi(texts, training_ids, params["vocab"])
    style = use_styles(texts, training_ids, params["style"])
    return np.array([np.concatenate((vocab[i], style[i])) for i in range(len(vocab))])


def style_lsi_separate(texts, training_ids, params):
    """Returns separate LSI vector and style vector for each text

    params should be a dict of two keys:
        - "vocab", containing the params you'd put into vectorise_lsi()
        - "style", containing the params you'd to into use_styles()
    """
    vocab = vectorise_lsi(texts, training_ids, params["vocab"])
    style = use_styles(texts, training_ids, params["style"])
    return vocab, style
