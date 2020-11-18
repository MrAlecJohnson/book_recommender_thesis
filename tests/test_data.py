import pandas as pd

cols3 = ["user_id", "book_id", "rating"]
cols4 = ["user_id", "book_id", "rating", "recommend"]

threshold_test = pd.DataFrame(
    [[0, 0, 1], [0, 1, 2], [1, 0, 3], [1, 1, 4], [2, 0, 5]], columns=cols3,
)

varied_ratings = pd.DataFrame(
    [
        [100, 10, 2, False],
        [100, 20, 5, True],
        [200, 10, 1, False],
        [200, 20, 4, True],
        [300, 10, 3, False],
        [300, 30, 5, True],
    ],
    columns=cols4,
)

similar_ratings = pd.DataFrame(
    [
        [0, 0, 2, False],
        [0, 1, 1, False],
        [1, 0, 1, False],
        [1, 1, 5, True],
        [2, 0, 5, True],
        [2, 1, 4, True],
    ],
    columns=cols4,
)

varied_review_counts = pd.DataFrame(
    [[0, 10, 2], [1, 20, 1], [1, 20, 1], [2, 30, 5], [2, 30, 5], [2, 30, 4]],
    columns=cols3,
)
