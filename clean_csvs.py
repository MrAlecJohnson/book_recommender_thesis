import csv

from pathlib import Path

CSV_DIR = Path.cwd() / "results"

# List of columns for results csvs
cols = [
    "description",
    "min_user_ratings",
    "min_book_ratings",
    "vectoriser",
    "vectoriser_params",
    "cf_algo",
    "cf_params",
    "content_algo",
    "content_params",
    "hybrid_algo",
    "hybrid_params",
    "time_taken",
    "fold",
    "cf_accuracy",
    "cf_f1",
    "content_accuracy",
    "content_f1",
    "hybrid_accuracy",
    "hybrid_f1",
    "cf_tp",
    "cf_tn",
    "cf_fp",
    "cf_fn",
    "cf_precision",
    "cf_recall",
    "content_tp",
    "content_tn",
    "content_fp",
    "content_fn",
    "content_precision",
    "content_recall",
    "hybrid_tp",
    "hybrid_tn",
    "hybrid_fp",
    "hybrid_fn",
    "hybrid_precision",
    "hybrid_recall",
]

# List of columns for results csvs that include separate style vectors
cols_style = [
    "description",
    "min_user_ratings",
    "min_book_ratings",
    "vectoriser",
    "vectoriser_params",
    "cf_algo",
    "cf_params",
    "content_algo",
    "content_params",
    "hybrid_algo",
    "hybrid_params",
    "time_taken",
    "fold",
    "cf_accuracy",
    "cf_f1",
    "content_accuracy",
    "content_f1",
    "style_accuracy",
    "style_f1",
    "hybrid_accuracy",
    "hybrid_f1",
    "cf_tp",
    "cf_tn",
    "cf_fp",
    "cf_fn",
    "cf_precision",
    "cf_recall",
    "content_tp",
    "content_tn",
    "content_fp",
    "content_fn",
    "content_precision",
    "content_recall",
    "style_tp",
    "style_tn",
    "style_fp",
    "style_fn",
    "style_precision",
    "style_recall",
    "hybrid_tp",
    "hybrid_tn",
    "hybrid_fp",
    "hybrid_fn",
    "hybrid_precision",
    "hybrid_recall",
]


def new_file(name, style):
    """Makes a csv of headers with a specified filename
    Returns nothing but creates a csv file

    Parameters
    ----------
    name (str):
        Filename to use for the file - added to the invariant CSV_DIR
        path specified at the start of this script

    style (bool):
        Whether this csv should include the extra columns for separate style vectors
    """
    with open(CSV_DIR / name, "w") as f:
        writer = csv.writer(f)
        if style:
            writer.writerow(cols_style)
        else:
            writer.writerow(cols)


# Create the standard 6 csv files
new_file("averaged_results.csv", False)
new_file("main_results.csv", False)
new_file("test_set_results.csv", False)
new_file("3_vector_averaged_results.csv", True)
new_file("3_vector_main_results.csv", True)
new_file("3_vector_test_set_results.csv", True)
