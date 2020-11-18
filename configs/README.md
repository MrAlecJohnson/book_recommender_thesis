# Config files
Whenever you run a group of experiments, 'config.json' will be used to set variables that remain the same for all experiments. The json should set these variables:

- **csv_main** (str): relative path to csv file - contains training results separated by cross-validation fold
- **csv_averages** (str): relative path to csv file - contains training results as an average of the cross-validation folds
- **csv_test_set** (str): relative path to csv file - contains test results
- **3_vector_main** (str): relative path to csv file - training results separated by cross-validation fold when using separate style and vocabulary vectors
- **3_vector_averages** (str): relative path to csv file - training results averaging cross-validation folds when using separate style and vocabulary vectors
- **3_vector_test_set** (str): relative path to csv file - test set results when using separate style and vocabulary vectors
- **min_user_ratings** (int): the minimum number of ratings each user must have made to be included in the dataset (10 is recommended)
- **min_book_ratings** (int): the minimum number of ratings each book must have to be included in the dataset (5 is recommended)
- **threshold** (int): the minimum rating to be considered a recommendation (4 is recommended)
- **random_state** (int): value to use wherever the code accepts a seed value for randomised values - for example test/train splits
- **min_proportion_positive** (float): any user whose proportion of positive ratings is less than this value will be left out of the dataset (0.2 is recommended)
- **max_proportion_positive** (float): any user whose proportion of positive ratings is greater than this value will be left out of the dataset (0.8 is recommended)
- **training_proportion** (float): proportion of the dataset to use as training data - the rest goes in the test set
- **training_run** (bool): if true, calculate results based on the test set. Otherwise just use the training data

See my project report for explanation of recommended values.
