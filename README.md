# A hybrid book recommender with stylistic elements

Alec Johnson's thesis project for the Birkbeck data science MSc. [Read the project report](https://docs.google.com/document/d/1hq0mDBW0hvx6sKy9imr8Gnmwju74MF0kpwdn1WDs-z4/edit?usp=sharing) on Google Docs.

Uses data from Project Gutenberg and Goodreads to explore algorithms for recommending books.


## Requirements
Python 3.6 or greater. Install necessary packages with `pip install -r requirements.txt`. This includes requirements for all sections of the project, including for data gathering and preprocessing.

See requirements.txt itself for details - you can leave some packages out if you just want to do the analysis (including leaving out the big Spacy model used in preprocessing).

To run the data acquisition notebooks you'll need a [Goodreads API key](https://www.goodreads.com/api).

A pre-commit config is included to help with further work on the repo. Set this up by [installing the pre-commit framework](https://pre-commit.com/) and then running `pre-commit install` from the project's root folder.


## Gathering and preparing the data
The notebooks folder contains notebooks for collecting and cleaning the Goodreads and Gutenberg data. Run the notebooks in order:

1. **read_rdf**: read metadata to make a list of English-language fiction from Project Gutenberg
2. **get_books**: download the English-language fiction from Project Gutenberg
3. **goodreads_data**: get book data from Goodreads and combine it with the Gutenberg metadata
4. **data_clean**: remove duplicates and inaccurately linked data from the previous step
5. **sqlite_setup**: make an empty database ready to receive user ratings
6. **goodreads_votes**: get user ratings from Goodreads

### Replicability
User ratings data isn't replicable because the data collection depends on Selenium session connections. See report for details.

Gutenberg data is mostly replicable but books are frequently added to the site. This means more recently run notebooks could find and download more books.


## Running the analysis
First, run the preprocessing on all your book texts: `python run_preprocessing.py`

Next make sure there's a config.json in the config folder. Create or choose a settings file that defines a set fo experiments. Then run:

`python run_experiments.py settings`

Where 'settings' is the name of the settings file. No need to include the '.json'.

### Config files
A config specifies csv files for results, and the parameters of the dataset to use in the experiments. These parameters cover:
- minimum and maximum user ratings per book
- rating threshold for considering a book recommended (usually set to 4 - see report)
- a random state for the entire run of experiments
- minimum and maximum proportions or positive ratings a user must give (see report)
- the proportion of data to use for the training set
- whether to run the experiments in training or test mode

See the config folder for details of writing these files.

### Settings files
A settings file is a json list, where each list element is a dictionary of experiment parameters that specify:
- a description of the experiment to list in the results
- a vectoriser and its parameters
- collaborative filtering algorithm and its parameters
- content-based filtering algorithm and its parameters
- hybridisation algorithm and its parameters

See the settings folder for details of writing these files.

### Tests
Run unit tests by running `pytest` from the project's root directory. There are currently tests for most of the modules. There aren't tests for classes or runners. See report for details - basically this comes down to a lack of time.

The `initial` files in the settings folder function as end-to-end tests, ensuring full experiments run for all the algorithms and vectorisers used in the project.


## Structure of the repo
The base folder contains 2 runner files:
- **run_preprocessing**: this processes every book in the texts folder and creates a processed output file for each one
- **run_experiments**: trains and tests a selection of different recommender algorithms and vectorisers and hybridisation techniques, as defined by a config file and a settings file

### Configs folder
This contains config files for running experiments. Only the file called 'config.json' is used by the main runner. Others are just for reference and for swapping in and out by changing filenames.

### Data folder
Book texts aren't included in the repo for size reasons. You can create them using the notebooks above, or arrange to get the data from me. The raw texts should be in a subfolder of the data folder. There should be another subfolder for the preprocessed versions of the texts.

For now I've included the book data (book_data_cut.csv) and the ratings data (book_ratings.db) in the data folder. I'll be deleting these after the project is marked.

### Models folder
Contains the pre-saved style vectors, the raw data for calculating those style vectors, and any pickle files from saved experiments.

### Modules folder
Code for the main functions and classes of the recommender system. Two Python files can be run directly:

- **clean_csvs**: this prepares 6 csv files ready to receive results. They are empty apart from their header rows. Warning: this will overwrite any existing files with the standard names.
- **extract_features**: trains and saves a set of vectors based on stylistic features. Runs on the raw text of all the books in the data folder.

The remaining files contain functions and classes used by the runner files in the root directory:

- **analyse**: cross-class functions for manipulating data
- **evaluate**: assess the results of an experiment
- **experiment_objects**: defines ExperimentData, Experiment and Fold
- **hybridise**: combine collaborative filtering and content-based vectors in various ways
- **load_data**: get and select database records to use in the analysis
- **preprocess**: manipulate book text in various ways
- **vectorise**: turn the text of a book into a numerical vector in various ways

### Notebooks folder
Notebooks containing code for acquiring and preparing data. These include the 6 numbered stages above, plus a test file you can ignore. There's also the notebook 'surprise', containing code for running Scikit Surprise on my dataset as a benchmark for collaborative filtering.

### Results folder
The csv files where experiment results are saved.

### Settings folder
Contains settings files, each defining a set of experiments to run.

### Tests folder
Contains unit tests for modules
