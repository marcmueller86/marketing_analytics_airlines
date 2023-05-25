import pandas as pd
import IPython
import seaborn as sns
from scipy import stats
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset containing sentiments, reviews, ratings, and topics
df = pd.read_csv('sentiments_reviews_ratings_topics.csv')

# Select specific airlines for analysis
airlines = ['ryanair', 'EasyJet', 'norwegian']
df = df[(df['airline'] == 'ryanair') | (df['airline'] == 'EasyJet') | (df['airline'] == 'norwegian')]

# Create dummy variables for sentiments
sentiment_dummies = pd.get_dummies(df['sentiment'])

# Select the variables for correlation analysis
variables = ['MONEY', 'PERSON', 'TIME', 'DATE', 'Value For Money', 'Rating Value',
             'Food & Beverages', 'Inflight Entertainment', 'Seat Comfort', 'Staff Service',
             'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6',
             'topic_7', 'topic_8', 'topic_9']

# Fill missing values in selected variables with 0
df[variables] = df[variables].fillna(0)

# Create an empty DataFrame to store correlation results
output_df = pd.DataFrame(columns=['airline', 'correlation_sentiment', 'correlation_entity', 'correlation_value'])

# Iterate over each airline
for airline in df['airline'].unique():
    # Filter the dataframe for the current airline
    airline_df = df[df['airline'] == airline]
    
    # Concatenate the sentiment dummies and variables for the current airline
    correlation_df = pd.concat([sentiment_dummies, airline_df[variables]], axis=1)
    
    # Calculate the correlation matrix for the current airline
    correlation_matrix = correlation_df.corr()
    
    # Iterate over sentiments and entities
    for sentiment in sentiment_dummies.columns:
        for entity in variables:
            # Get the correlation value for the sentiment-entity pair
            correlation = correlation_matrix[sentiment][entity]
            
            # Append the correlation results to the output dataframe
            output_df = output_df.append({
                'airline': airline,
                'correlation_sentiment': sentiment,
                'correlation_entity': entity,
                'correlation_value': correlation
            }, ignore_index=True)

# Save the correlation results to a CSV file
output_df.to_csv('sentiments_reviews_ratings_topics_correlation.csv', index=False)

# Prepare the dataset for regression analysis
df['sentiment_target_negative'] = df['sentiment'].apply(lambda x: 1 if x == 'sentiment_negative' else 0)
df['sentiment_target_positive'] = df['sentiment'].apply(lambda x: 1 if x == 'sentiment_positive' else 0)
df['sentiment_target_neutral'] = df['sentiment'].apply(lambda x: 1 if x == 'sentiment_neutral' else 0)

# Define the features and target variables for regression analysis
features = ['MONEY', 'PERSON', 'TIME', 'DATE', 'Value For Money', 'Rating Value',
            'Food & Beverages', 'Inflight Entertainment', 'Seat Comfort',
            'Staff Service', 'topic_0', 'topic_1', 'topic_2', 'topic_3',
            'topic_4', 'topic_5', 'topic_6', 'topic_7', 'topic_8', 'topic_9']
targets = ['sentiment_target_positive', 'sentiment_target_neutral', 'sentiment_target_negative']

# Create a DataFrame to store the regression analysis results
df_res = pd.DataFrame()

# Iterate over each airline and target variable
for airline in airlines:
    for target in targets:
        print('###################### MODELS {} for target {} \n'.format(airline, target))
        
        # Filter the dataframe for the current airline
        df_airline = df[df['airline'] == airline]
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(df_airline[features], df_airline[target], test_size=0.2, random_state=42)

        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the linear regression model
        mse = mean_squared_error(y_test, y_pred)
        print('Linear Regression {} for target {}'.format(airline, target))
        print("Mean Squared Error:", mse)

        # Calculate R-squared
        r2 = r2_score(y_test, y_pred)
        print("R-squared:", r2)

        # Get the coefficients (importance) of each feature
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.coef_})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        feature_importance['type'] = 'linear'
        feature_importance['prediction'] = target
        feature_importance_linear = feature_importance

        ##### Logistic Regression

        # Create and fit the logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print('Logistic Regression {} for target {} \n'.format(airline, target))
        print("Accuracy:", accuracy)

        # Calculate F1 score
        f1 = f1_score(y_test, y_pred)
        print("F1 Score:", f1)

        # Get the coefficients (importance) of each feature
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.coef_[0]})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        feature_importance['type'] = 'logistic'
        feature_importance['prediction'] = target
        feature_importance_logistic = feature_importance

        # Create confusion matrix
        confusion_mat = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(confusion_mat)

        # Create and fit the XGBoost model
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        print('XGBoost {} for target {} \n'.format(airline, target))
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Get feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        feature_importance['type'] = 'xgboost'
        feature_importance_xgboost = feature_importance

        print(feature_importance)

        # Create confusion matrix
        confusion_mat = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(confusion_mat)

        feature_importance_xgboost['prediction'] = target

        # Store the feature importance results for each model type
        feature_importance_dfs = [df_res, feature_importance_xgboost, feature_importance_logistic, feature_importance_linear]

        # Concatenate the feature importance dataframes
        concatenated_df = pd.concat(feature_importance_dfs)

        # Reset the index of the concatenated dataframe
        concatenated_df = concatenated_df.reset_index(drop=True)

# Save the feature importance results to a CSV file
concatenated_df.to_csv('feature_importance_sentiment.csv', index=False)
