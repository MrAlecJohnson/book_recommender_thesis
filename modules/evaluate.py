import csv
import numpy as np

from pathlib import Path


def count_true(df, col_to_check):
    """Counts true positives, true negatives, false positives and false negatives
    Returns tuple of these 4 counts

    Parameters
    ----------
    df (DataFrame):
        DataFrame of results with the target values in a column called 'true_class'

    col_to_check (str):
        Name of the column to compare to true_class
    """
    tp = sum((df["true_class"]) & (df[col_to_check]))
    tn = sum((~df["true_class"]) & (~df[col_to_check]))
    fp = sum((~df["true_class"]) & (df[col_to_check]))
    fn = sum((df["true_class"]) & (~df[col_to_check]))
    return (tp, tn, fp, fn)


def stats(tp, tn, fp, fn):
    """Calculate precision, recall, f1 and accuracy from output of count_true
    Returns list of rounded values

    Parameters
    ----------
    tp, tn, fp, fn (int):
        Counts of true positive, true negative, false positive and false negative
    """
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * tp) / ((2 * tp) + fp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return [round(measure, 4) for measure in [precision, recall, f1, accuracy]]


def evaluate_results(results):
    """Runs count_true() and stats() for results from collaborative filtering,
    vocabulary-based filtering, hybridisation and, if included, style-based filtering.
    Then assembles those results into a list

    Parameters
    ----------
    results (DataFrame):
        Must contain columns for cf_class, content_class, hybrid_class and true_class.
        Optionally include a column for style_class
    """
    cf_tp, cf_tn, cf_fp, cf_fn = count_true(results, "cf_class")
    cf_precision, cf_recall, cf_f1, cf_accuracy = stats(cf_tp, cf_tn, cf_fp, cf_fn)

    content_tp, content_tn, content_fp, content_fn = count_true(
        results, "content_class"
    )
    content_precision, content_recall, content_f1, content_accuracy = stats(
        content_tp, content_tn, content_fp, content_fn
    )

    hybrid_tp, hybrid_tn, hybrid_fp, hybrid_fn = count_true(results, "hybrid_class")
    hybrid_precision, hybrid_recall, hybrid_f1, hybrid_accuracy = stats(
        hybrid_tp, hybrid_tn, hybrid_fp, hybrid_fn
    )

    scores = [
        cf_accuracy,
        cf_f1,
        content_accuracy,
        content_f1,
        hybrid_accuracy,
        hybrid_f1,
        cf_tp,
        cf_tn,
        cf_fp,
        cf_fn,
        cf_precision,
        cf_recall,
        content_tp,
        content_tn,
        content_fp,
        content_fn,
        content_precision,
        content_recall,
        hybrid_tp,
        hybrid_tn,
        hybrid_fp,
        hybrid_fn,
        hybrid_precision,
        hybrid_recall,
    ]

    if "style_class" in results:
        style_tp, style_tn, style_fp, style_fn = count_true(results, "style_class")
        style_precision, style_recall, style_f1, style_accuracy = stats(
            style_tp, style_tn, style_fp, style_fn
        )
        scores[4:4] = [
            style_accuracy,
            style_f1,
        ]
        scores[20:20] = [
            style_tp,
            style_tn,
            style_fp,
            style_fn,
            style_precision,
            style_recall,
        ]

    return scores


def average_folds(fold_scores):
    """Elementwise average of an array of lists, returned as a list
    Used for taking the average of cross-validation
    """
    arrays = np.array(fold_scores)
    return np.mean(arrays, axis=0).tolist()


def write_results(config, experiment, time):
    """Writes experiment output to spreadsheets. Includes scores from
    evaluate_results() plus the parameters of the experiment

    Returns nothing but adds rows to csvs depending on the type of experiment.

    Parameters
    ----------
    config (dict):
        Dictionary of config settings

    experiment (Experiment):
        An experiment object that's previously run evaluate_test() or evaluate_folds()

    time (numeric):
        Float or int representing time since the experiment began
    """
    # List of descriptive parameters to get from config and Experiment
    settings = [
        experiment.description,
        config["min_user_ratings"],
        config["min_book_ratings"],
        experiment.vectoriser,
        experiment.vectoriser_params,
        experiment.cf_algo,
        experiment.cf_params,
        experiment.content_algo,
        experiment.content_params,
        experiment.hybrid_algo,
        experiment.hybrid_params,
        round(time),
    ]

    # Training runs add cross-validation results to one csv and averages to another
    # Different csvs for results that include separate style vectors
    if config["training_run"]:
        if experiment.vectoriser == "separate":
            main = Path.cwd() / config["3_vector_main"]
            avg = Path.cwd() / config["3_vector_averages"]
        else:
            main = Path.cwd() / config["csv_main"]
            avg = Path.cwd() / config["csv_averages"]

        with open(main, "a") as f:
            writer = csv.writer(f)
            for i in range(len(experiment.fold_scores)):
                row = settings + [i] + experiment.fold_scores[i]
                writer.writerow(row)
        with open(avg, "a") as f:
            writer = csv.writer(f)
            row = settings + ["Average"] + experiment.scores
            writer.writerow(row)

    # Test runs just add to the test results spreadsheet
    # Still has different csv for results that include separate style vectors
    else:
        if experiment.vectoriser == "separate":
            test = Path.cwd() / config["3_vector_test_set"]
        else:
            test = Path.cwd() / config["csv_test_set"]

        with open(test, "a") as f:
            writer = csv.writer(f)
            row = settings + ["Test"] + experiment.scores
            writer.writerow(row)
