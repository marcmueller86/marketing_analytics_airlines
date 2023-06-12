import pandas as pd
import IPython
import seaborn as sns
from scipy import stats
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz

import pydotplus
import re

import matplotlib.pyplot as plt
import graphviz
import json
from itertools import combinations
from collections import Counter

# Helper function to extract sample count and entropy from the node label
def get_leaf_info(tree, feature_names, node_id=0):
    feature_index = tree.feature[node_id]
    feature = feature_names[feature_index]

    threshold = tree.threshold[node_id]
    samples = tree.n_node_samples[node_id]
    value = tree.value[node_id][0]
    class_prob = value / np.sum(value)
    entropy = -np.sum(class_prob * np.log2(class_prob + 1e-10))

    if tree.children_left[node_id] == tree.children_right[node_id]:
        return {
            'feature': feature,
            'threshold': float(threshold),
            'samples': int(samples),
            'class_prob': class_prob.tolist(),
            'entropy': float(entropy)
        }

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    return {
        'feature': feature,
        'threshold': float(threshold),
        'samples': int(samples),
        'class_prob': class_prob.tolist(),
        'entropy': float(entropy),
        'left': get_leaf_info(tree, feature_names, left_child),
        'right': get_leaf_info(tree, feature_names, right_child)
    }



def get_top_combinations(group):
    from itertools import combinations
    from collections import Counter

    topics = group[['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6',
             'topic_7', 'topic_8', 'topic_9']]
    topic_combinations = []
    for _, row in topics.iterrows():
        topic_list = [topic for topic in row.index if row[topic] == 1]
        topic_combinations.extend(combinations(topic_list, 2))
    top_combinations = Counter(topic_combinations).most_common(10)
    return top_combinations

# Load the dataset containing sentiments, reviews, ratings, and topics
df = pd.read_csv('sentiments_reviews_ratings_topics.csv')

# Select specific airlines for analysis
airlines = ['ryanair', 'EasyJet', 'norwegian']
df = df[(df['airline'] == 'ryanair') | (df['airline']
                                        == 'EasyJet') | (df['airline'] == 'norwegian')]

# Create dummy variables for sentiments
sentiment_dummies = pd.get_dummies(df['sentiment'])
# Select the variables for correlation analysis
variables = ['MONEY', 'PERSON', 'TIME', 'DATE', 'Value For Money', 'Rating Value',
             'Food & Beverages', 'Inflight Entertainment', 'Seat Comfort', 'Staff Service',
             'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6',
             'topic_7', 'topic_8', 'topic_9']

# Fill missing values in selected variables with 0
df[variables] = df[variables].fillna(0)

metric_frame = []
# Create an empty DataFrame to store correlation results
output_df = pd.DataFrame(columns=[
                         'airline', 'correlation_sentiment', 'correlation_entity', 'correlation_value'])
# Iterate over each airline
for airline in df['airline'].unique():
    # Filter the dataframe for the current airline
    airline_df = df[df['airline'] == airline]

    # Concatenate the sentiment dummies and variables for the current airline
    correlation_df = pd.concat(
        [sentiment_dummies, airline_df[variables]], axis=1)

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
output_df.to_csv(
    'sentiments_reviews_ratings_topics_correlation.csv', index=False)

# Prepare the dataset for regression analysis
df['sentiment_target_negative'] = df['sentiment'].apply(
    lambda x: 1 if x == 'sentiment_negative' else 0)
df['sentiment_target_positive'] = df['sentiment'].apply(
    lambda x: 1 if x == 'sentiment_positive' else 0)
df['sentiment_target_neutral'] = df['sentiment'].apply(
    lambda x: 1 if x == 'sentiment_neutral' else 0)

# Define the features and target variables for regression analysis
features = ['MONEY', 'PERSON', 'TIME', 'DATE', 'Value For Money', 'Rating Value',
            'Food & Beverages', 'Inflight Entertainment', 'Seat Comfort',
            'Staff Service', 'topic_0', 'topic_1', 'topic_2', 'topic_3',
            'topic_4', 'topic_5', 'topic_6', 'topic_7', 'topic_8', 'topic_9']
targets = ['sentiment_target_positive',
           'sentiment_target_neutral', 'sentiment_target_negative']

# Create a DataFrame to store the regression analysis results
df_res = pd.DataFrame()
feature_importance_dfs_all = []
# Iterate over each airline and target variable
for airline in airlines:
    for target in targets:
        print('###################### MODELS {} for target {} \n'.format(
            airline, target))

        # Filter the dataframe for the current airline
        df_airline = df[df['airline'] == airline]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df_airline[features], df_airline[target], test_size=0.2, random_state=42)

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
        feature_importance = pd.DataFrame(
            {'Feature': features, 'Importance': model.coef_})
        feature_importance = feature_importance.sort_values(
            by='Importance', ascending=False)
        feature_importance['type'] = 'linear'
        feature_importance['prediction'] = target
        feature_importance['airline'] = airline
        feature_importance_linear = feature_importance

        metric_frame.append(
            {
                'Mean Squared Error': mse,
                'R-squared': r2,
                'algorithm': 'Linear Regression',
                'airline': airline,
                'count_samples': len(y_test),
                'prediction': target
            }
        )


        # Logistic Regression

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
        feature_importance = pd.DataFrame(
            {'Feature': features, 'Importance': model.coef_[0]})
        feature_importance = feature_importance.sort_values(
            by='Importance', ascending=False)
        feature_importance['type'] = 'logistic'
        feature_importance['prediction'] = target
        feature_importance['airline'] = airline
        feature_importance_logistic = feature_importance

        # Create confusion matrix
        confusion_mat = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(confusion_mat)

        metric_frame.append(
            {
                'f1': f1,
                'Accuracy': accuracy,
                'algorithm': 'Logistic Regression',
                'airline': airline,
                'count_samples': len(y_test),
                'prediction': target,
                'True positive': confusion_mat[0][0],
                'False positive': confusion_mat[1][0],
                'False negative': confusion_mat[0][1],
                'True negative': confusion_mat[1][1]
            }
        )

        # Create and fit the XGBoost model

        model = xgb.XGBClassifier(max_depth=4)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        print('XGBoost {} for target {} \n'.format(airline, target))
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Get feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame(
            {'Feature': features, 'Importance': importance})
        feature_importance = feature_importance.sort_values(
            by='Importance', ascending=False)
        feature_importance['type'] = 'xgboost'
        feature_importance['airline'] = airline
        feature_importance['prediction'] = target

        feature_importance_xgboost = feature_importance

        print(feature_importance)

        # Create confusion matrix
        confusion_mat = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(confusion_mat)

        metric_frame.append(
            {
                'f1': f1,
                'Accuracy': accuracy,
                'algorithm': 'XGBoost',
                'airline': airline,
                'count_samples': len(y_test),
                'prediction': target,
                'True positive': confusion_mat[0][0],
                'False positive': confusion_mat[1][0],
                'False negative': confusion_mat[0][1],
                'True negative': confusion_mat[1][1]
            }
        )


        # Store the feature importance results for each model type
        feature_importance_dfs = [df_res, feature_importance_xgboost,
                                  feature_importance_logistic, feature_importance_linear]
        feature_importance_dfs_all.extend(feature_importance_dfs)

        # Create and fit the Decision Tree model
        model = DecisionTreeClassifier(max_depth=4,criterion='entropy',min_samples_leaf=20)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        print('Decision Tree {} for target {} \n'.format(airline, target))
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Get feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        feature_importance['type'] = 'decision_tree'
        feature_importance['airline'] = airline
        feature_importance['prediction'] = target

        feature_importance_decision_tree = feature_importance

        print(feature_importance)

        # Create confusion matrix
        confusion_mat = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(confusion_mat)

        metric_frame.append({
            'f1': f1,
            'Accuracy': accuracy,
            'algorithm': 'Decision Tree',
            'airline': airline,
            'count_samples': len(y_test),
            'prediction': target,
            'True positive': confusion_mat[0][0],
            'False positive': confusion_mat[1][0],
            'False negative': confusion_mat[0][1],
            'True negative': confusion_mat[1][1]
        })
        # Plot the decision tree
        class_names = ['false','true']

        plt.figure(figsize=(10, 10))
        plot_tree(model, feature_names=features, class_names=class_names, filled=True, rounded=True)
        # Customize node colors
        colors = np.where(model.tree_.value[:, 0, 0] > model.tree_.value[:, 0, 1], 'lightgreen', 'salmon')
        plt.rcParams['text.color'] = 'black'
        plt.rcParams['axes.labelcolor'] = 'black'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'

        # Apply node colors to the plot
        for node, color in zip(model.tree_.children_left, colors):
            if node != -1:
                box = plt.gca().get_children()[node].get_bbox_patch()
                box.set_facecolor(color)

        # Add class probabilities to the last leaf
        leaf_id = np.where(model.tree_.children_left == -1)[0][-1]
        class_probs = model.tree_.value[leaf_id][0] / np.sum(model.tree_.value[leaf_id][0])
        class_probs_str = ', '.join(f'{class_prob:.2f}' for class_prob in class_probs)
        text = f'Class Probs: [{class_probs_str}]'
        x, y, _, _ = plt.gca().get_children()[leaf_id].get_bbox_patch().get_extents().bounds
        plt.text(
            x + 0.5,
            y + 0.5,
            text,
            fontsize=10,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        )
        plt.title('{} - {}.png'.format(airline,target))
        plt.savefig('dt_{}_{}.png'.format(airline,target),dpi=300)


        # Get feature importances and names
        feature_importances = model.feature_importances_
        feature_names = list(X_train.columns)
        print (feature_names)
        # Sort feature names based on importances
        sorted_features = [feature for _, feature in sorted(zip(feature_importances, feature_names), reverse=True)]

        # Get leaf information from the decision tree
        leaf_info = get_leaf_info(model.tree_, sorted_features)


        # Save tree JSON to a file
        with open('dt_{}_{}.json'.format(airline,target), 'w') as json_file:
            json.dump(leaf_info, json_file)
        # Save tree JSON to a file

# Concatenate the feature importance dataframes
concatenated_df = pd.concat(feature_importance_dfs_all, axis=0)

# Reset the index of the concatenated dataframe
concatenated_df = concatenated_df.reset_index(drop=True)


# Save the feature importance results to a CSV file
concatenated_df.to_csv('feature_importance_sentiment.csv', index=False)
# import IPython
# IPython.embed()
df_metrics = pd.DataFrame(metric_frame)
df_metrics = df_metrics.fillna(0)
df_metrics.to_csv('metrics_features_algorithms.csv',float_format='%.3f', index=False)


### Create Topics Aggregation: 

topics = ['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4',
       'topic_5', 'topic_6', 'topic_7', 'topic_8', 'topic_9',]
df_all_topics = []
for topic in topics:
    # Pivot elements
    pivot_elements = ['published_time', 'airline', 'sentiment', topic]

    # Columns to calculate average
    average_columns = ['Food & Beverages', 'Inflight Entertainment', 'Seat Comfort', 'Staff Service',
                    'Value for Money', 'Type Of Traveller', 'Seat Type', 'Route', 'Date Flown',
                    'Cabin Staff Service', 'Ground Service', 'Wifi & Connectivity', 'Value For Money']

    # Columns to calculate percentage
    percentage_columns = ['Recommended', 'MONEY', 'TIME', 'PERSON', 'DATE']

    # Calculate average
    df_avg = df.groupby(pivot_elements)[average_columns].mean().reset_index()

    # Calculate percentage
    df_percentage = df.groupby(pivot_elements)[percentage_columns].apply(lambda x: (x == 1).mean()).reset_index()

    # Merge the average and percentage DataFrames
    df_transformed = pd.merge(df_avg, df_percentage, on=pivot_elements)

    # Count rows
    df_count = df.groupby(pivot_elements).size().reset_index(name='Count')

    # Merge count with transformed DataFrame
    df_transformed = pd.merge(df_transformed, df_count, on=pivot_elements)
    df_transformed_filtered = df_transformed[df_transformed[topic] == 1]
    df_transformed_filtered = df_transformed_filtered.rename(columns={topic: 'topic'})
    df_transformed_filtered['topic'] = topic

    # Print the transformed DataFrame with count
    df_all_topics.append(df_transformed_filtered)
df_all_topics_export = pd.concat(
    df_all_topics, axis=0)
df_all_topics_export.to_csv('topics_aggregatted.csv',float_format='%.3f', index=False)


### significant test
from scipy.stats import chi2_contingency

columns_of_interest = ['MONEY', 'PERSON', 'TIME', 'DATE', 'Value For Money', 'Rating Value',
                       'Food & Beverages', 'Inflight Entertainment', 'Seat Comfort', 'Staff Service',
                       'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6',
                       'topic_7', 'topic_8', 'topic_9']

# Perform the chi-square test for each column
total_rows = len(df)
columns_to_change = ['Value For Money', 'Rating Value', 'Food & Beverages', 'Inflight Entertainment', 'Seat Comfort', 'Staff Service']
df_chi = df
for column in columns_to_change:
    df_chi[column] = df_chi[column].apply(lambda x: 1 if x > 0 else x)
res = []
# Perform the chi-square test for each column
for column in columns_of_interest:
    observed_freq = df_chi[column].value_counts().values
    # Create a hypothetical uniform distribution with expected frequencies
    expected_freq = np.full(len(observed_freq), total_rows / len(observed_freq))
    # Perform the chi-square test
    chi2, p_value = chi2_contingency([observed_freq, expected_freq])[0:2]
    res.append({"Column": column, "Chi-square statistic": chi2, "p-value": p_value, 'significant': p_value < 0.05 })
    print("Column:", column)
    print("Chi-square statistic:", chi2)
    print("p-value:", p_value)
    is_significant = p_value < 0.05  # Assuming significance level of 0.05
    print("Significant:", is_significant)


df_chi_res = pd.DataFrame(res).to_csv('significant_test.csv',index=False)

# basket case analysis
results = pd.DataFrame(columns=['airline', 'sentiment', 'top_combinations'])

grouped = df.groupby(['airline', 'sentiment'])

for group_name, group in grouped:
    airline, sentiment = group_name
    top_combinations = get_top_combinations(group)
    results = results.append({'airline': airline, 'sentiment': sentiment, 'top_combinations': top_combinations}, ignore_index=True)

print(results)
results.to_csv('basket_case_topics.csv',index=False)


