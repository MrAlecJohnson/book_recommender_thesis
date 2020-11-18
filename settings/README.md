# Settings files
When you use run_experiments.py you should specify a settings file from this folder as a command line parameter. The settings file should define one or more experiments to run, using json format.

## Structure of the settings file
The outermost layer should be a list, with each experiment defined as a dictionary in that list. Each dictionary must set:

- **description** (str): how to describe the experiment in the results file
- **vectoriser** (str): the technique used to convert book text into content vectors
- **vectoriser_params** (dict): parameters for the vectoriser
- **cf_algo** (str): the algorithm used in collaborative filtering
- **cf_params** (dict): parameters for the collaborative filtering algorithm
- **content_algo** (str): the algorithm used in content-based recommendation
- **content_params** (dict): parameters for the content-based algorithm
- **hybrid_algo** (str): the hybrid used to combine the collaborative filtering and content-based recommenders
- **hybrid_params** (dict): parameters for the hybrid

## Vectoriser options
Parameters for vocabulary-based vectorisers come from the linked Scikit Learn implementations.
- "count": vectors of [term counts](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- "tfidf": vectors of [term frequency/inverse document frequency](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- "lsi": latent semantic indexing - parameters should look like: {"normalise": true, "tfidf": {params for tfidf}, "lsi": {[params from Scikit Learn truncated SVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)}}
- "style": style-based vectors - parameters should be {"location": "models/features.pkl", "normalise": true}
- "random": vectors of random numbers between 0 and 1 - parameters should look like: {"random_state": 50, "length": 100}
- "combined": creates vocabulary and style vectors then concatenates them into one - parameters should look like: {"vocab": {parameters for any vocab vector}, "style": {parameters as for 'style' vector}},
- "separate": creates vocabulary and style vectors and returns them separately - parameters should look like: {"vocab": {parameters for any vocab vector}, "style": {parameters as for 'style' vector}},


## Algorithm options
### For all algorithm choices
Parameters for these algorithms should match their linked Scikit Learn implementations.
- "naive_bayes": [multinomial naive Bayes classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)
- "knn": [k-nearest neighbours classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- "decision_tree": [decision tree classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- "svc": [support vector classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- "log_regression": [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Hybrid only
These rule-based hybrids don't have any parameters.
- "confidence": picks whichever algorithm is most confident
- "book_rating": uses book's average rating as a tie-breaker
- "best_of_3": only works when vectoriser is set to "separate" - takes the best of 3 out of collaborative filtering, vocabulary vector recommendation and style vector recommendation

## Example file
A complete settings file defining a single experiment could look like this:

``` json
[
    {
        "description": "Naive bayes for all three",
        "vectoriser": "tfidf",
        "vectoriser_params": {
                "analyzer": "word",
                "ngram_range": [1, 1],
                "min_df": 2,
                "max_features": 10000,
                "lowercase": false
        },
        "cf_algo": "naive_bayes",
        "cf_params": {
            "fit_prior": false
        },
        "content_algo": "naive_bayes",
        "content_params": {
            "fit_prior": false
        },
        "hybrid_algo": "naive_bayes",
        "hybrid_params": {
            "fit_prior": false
        }
    }
]
```
