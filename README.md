# Structure of code and files 
- src
   - crawler.py -> crawling data from skytrex and do a first processing
   - topic_moddling.py -> applying different topic moddeling approaches do a processing and joing of topics 
   - analytics.py -> applying different statistic and machine learning processes to the data and apply them 
- results 
   - tree
      - output of all dt of the three predictions 
   - json
      - output of the dt as nested json 

# Results

Results of the code can be fined in this looker analysis: 

<https://lookerstudio.google.com/s/uxmm2CWLWz8>

# Explanation of the analysis

`analytics.py`

The provided code performs analysis and modeling tasks on a dataset containing sentiments, reviews, ratings, and topics related to different airlines. Let's break down the code and explain its purpose and the reasons for the choices made:

1. **Data Loading and Filtering:**
   - The code begins by importing necessary libraries and reading the data from a CSV file using pandas (`pd.read_csv`).
   - The dataset is then filtered to include only the specified airlines (Ryanair, EasyJet, and Norwegian) using boolean indexing.

2. **Correlation Analysis:**
   - Dummy variables are created for the "sentiment" column using `pd.get_dummies` to represent sentiments as binary indicators (0 or 1).
   - A list of variables/features is defined for correlation analysis.
   - The variables in the dataset are filled with 0 if they contain missing values.
   - An empty DataFrame called `output_df` is created to store the correlation results.
   - The code iterates over each unique airline in the dataset.
   - For each airline, a correlation matrix is calculated for the sentiments and variables using `correlation_df.corr()`.
   - The code then iterates over each sentiment and variable to extract the correlation value between them and appends the results to `output_df`.
   - Finally, the `output_df` DataFrame is saved to a CSV file.

3. **Classification Analysis (Feature Importance):**
   - Three target variables for sentiment classification are defined: 'sentiment_target_positive', 'sentiment_target_neutral', and 'sentiment_target_negative'.
   - A new DataFrame called `df_res` is created to store the results of the regression analysis.
   - The code iterates over each airline and target variable combination.
   - For each combination, the dataset is filtered for the current airline.
   - The data is split into training and test sets using `train_test_split`.
   - Linear regression, logistic regression, and XGBoost models are created, trained, and evaluated on the test set.
   - Evaluation metrics such as mean squared error, R-squared, accuracy, F1 score, and confusion matrix are calculated and printed.
   - Feature importance is extracted from each model, and the results are stored in separate DataFrames (`feature_importance_linear`, `feature_importance_logistic`, `feature_importance_xgboost`).
   - The feature importance DataFrames are concatenated along with `df_res`.
   - Finally, the concatenated DataFrame is saved to a CSV file.

Reasons for Choices:
- **Correlation Analysis:** The code calculates the correlation between sentiments and various variables/features to understand their relationships. This analysis helps identify potential patterns and associations between sentiments and specific aspects of airline reviews.
- **Classification Analysis:** By performing regression analysis using linear regression, logistic regression, and XGBoost models, the code aims to predict the sentiment categories based on the given features. This can provide insights into the importance of different features in determining sentiment and compare the performance of different models.

Overall, the code provides a comprehensive analysis of sentiment, feature correlation, and sentiment prediction for different airlines, enabling insights into customer opinions and potential factors influencing sentiment.

# Explanation of the topic modeling

`crawler.py`

The provided code performs topic modeling on airline reviews using various algorithms such as Latent Dirichlet Allocation (LDA), Latent Semantic Analysis (LSA), Hierarchical Dirichlet Process (HDP), and Non-Negative Matrix Factorization (NMF). Here's a breakdown of the code and its functionalities:

1. Importing Required Libraries:
   - The code starts by importing the necessary libraries, including pandas, scikit-learn's vectorizers (CountVectorizer, TfidfVectorizer, HashingVectorizer), decomposition models (LatentDirichletAllocation, TruncatedSVD, NMF), and the Pipeline class.

2. Loading Data:
   - The code loads the airline reviews from a CSV file ('export_airlines_complete.csv') into a pandas DataFrame called `df`.

3. Defining Stopwords:
   - A list of stopwords is defined, containing common words that are typically irrelevant in text analysis.

4. Initializing Variables and DataFrames:
   - The code initializes variables such as `airlines` (a list of airline names) and an empty DataFrame `df_res` to store the results.

5. Looping over Airlines:
   - The code iterates over each airline in the `airlines` list.

6. Filtering Data for Each Airline:
   - A temporary DataFrame `df_temp` is created by filtering the original DataFrame `df` for the current airline.

7. Checking if Data is Empty:
   - If `df_temp` is empty (i.e., no reviews for the current airline), the code skips to the next iteration.

8. Vectorization:
   - A TfidfVectorizer is defined with a maximum of 500 features, lowercase text, and using the stopwords defined earlier.
   - The vectorizer is fitted on the review texts (`df_temp['review_text']`) to transform them into numerical feature vectors (`X`).
   - The feature names are retrieved using `get_feature_names_out()`.

9. Defining Topic Modeling Algorithms:
   - Several topic modeling algorithms are defined with specified parameters:
     - LatentDirichletAllocation (LDA) with 10 topics.
     - TruncatedSVD (LSA) with 10 components.
     - LatentDirichletAllocation (HDP) with 10 topics and additional parameters.
     - NMF with 10 components.

10. Creating Pipelines:
    - Pipelines are created for each algorithm, consisting of the vectorizer and the respective topic modeling model.

11. Fitting Pipelines:
    - Each pipeline is fitted on the review texts (`df_temp['review_text']`).

12. Extracting Topic-Word Distributions:
    - The topic-word distributions for each algorithm are obtained using the fitted models.

13. Printing Topics:
    - The top words for each topic are printed for each algorithm using the topic-word distributions and feature names.

14. Extracting Document-Topic Distributions:
    - The pipelines are fitted on the review texts to obtain the document-topic distributions.
    - The document-topic distributions are stored in variables `lda_topics`, `lsa_topics`, `hdp_topics`, and `nmf_topics`.

15. Adding Topics to DataFrame:
    - The topics are added to the temporary DataFrame `df_temp` by concatenating the document-topic distributions as new columns.
    - The topic values are binarized, where a value of 1 indicates the presence of the topic in the review.

16. Counting Topics per Review:
    - The number of topics per review is calculated by summing the binary topic values across topics and storing the result in the column `num_topics`.

17. Concatenating DataFrames:
    - The temporary DataFrame `df_temp` is concatenated with the result DataFrame `df_res` using `pd.concat()`
