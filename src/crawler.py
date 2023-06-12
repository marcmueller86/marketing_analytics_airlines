import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
import IPython
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.parse.corenlp import CoreNLPParser
import spacy
import matplotlib.pyplot as plt
from bs4 import Comment
import seaborn as sns
from scipy import stats
import numpy as np
from datetime import datetime



# 6 airlines vergleichen 
# empfehlungen was mögen die kunden
# was mögen die kunden von lufthansa was kann optimiert werden 
# eine airline im vergleich 
# Handlungsempfehlungen an die XYZ Airline auf Grundlage der Analyse der Kundenbewertungen im Vergleich mit ABC
# ein segment auswählen
# eurowings, easyjet, ryanair, Jetblue Airways, Westjet	
# vergleich ryanair 

# Send a GET request to the URL
urls = [
       {"name": "norwegian","url":"https://www.airlinequality.com/airline-reviews/norwegian/?sortby=post_date%3ADesc&pagesize=2500","url_pagination":"https://www.airlinequality.com/airline-reviews/norwegian/page/{page}/?sortby=post_date%3ADesc&pagesize=100"}, 
       {"name": "EasyJet","url":"https://www.airlinequality.com/airline-reviews/EasyJet/?sortby=post_date%3ADesc&pagesize=2500","url_pagination":"https://www.airlinequality.com/airline-reviews/EasyJet/page/{page}/?sortby=post_date%3ADesc&pagesize=100"},
       {"name": "ryanair","url":"https://www.airlinequality.com/airline-reviews/ryanair/?sortby=post_date%3ADesc&pagesize=2500","url_pagination":"https://www.airlinequality.com/airline-reviews/ryanair/page/{page}/?sortby=post_date%3ADesc&pagesize=100"},
       {"name": "wizz-air","url":"https://www.airlinequality.com/airline-reviews/wizz-air/?sortby=post_date%3ADesc&pagesize=2500","url_pagination":"https://www.airlinequality.com/airline-reviews/wizz-air/page/{page}/?sortby=post_date%3ADesc&pagesize=100"},
       {"name": "eurowings","url":"https://www.airlinequality.com/airline-reviews/eurowings/?sortby=post_date%3ADesc&pagesize=2500","url_pagination":"https://www.airlinequality.com/airline-reviews/eurowings/page/{page}/?sortby=post_date%3ADesc&pagesize=100"},
       {"name": "vueling-airlines","url":"https://www.airlinequality.com/airline-reviews/vueling-airlines/?sortby=post_date%3ADesc&pagesize=2500","url_pagination":"https://www.airlinequality.com/airline-reviews/vueling-airlines/page/{page}/?sortby=post_date%3ADesc&pagesize=100"},
       ]
# https://www.airlinequality.com/airline-reviews/norwegian/page/2/?sortby=post_date%3ADesc&pagesize=100



#
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('vader_lexicon')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Download the language model
spacy.cli.download("en_core_web_sm")

# Load the language model
nlp = spacy.load("en_core_web_sm")
if not nlp.has_pipe('ner'):
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)

result = []



def extract_entities(text):
    """
    Extracts named entities from the specified text.

    Parameters:
        text (str): The text to analyze.

    Returns:
        A list of named entity spans.
    """
    # Apply named entity recognition to the text using Spacy
    doc = nlp(text)
    named_entities = [(ent.label_, ent.text) for ent in doc.ents]

    # Filter the entities to include only those of interest
    named_entities = [(label, text) for label, text in named_entities 
                    if label in ['ORG', 'PERSON', 'GPE', 'MONEY', 'DATE', 'TIME']]
    return named_entities

def get_sentiment(text):
    """
    Computes the sentiment score for the specified text.

    Parameters:
        text (str): The text to analyze.

    Returns:
        A dictionary containing the sentiment scores.
    """
    # Compute the sentiment score for the text
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)

    # Map the score values to sentiment labels
    sentiment = {}
    if scores['compound'] >= 0.05:
        sentiment['sentiment_positive'] = scores['compound']
    elif scores['compound'] <= -0.05:
        sentiment['sentiment_negative'] = scores['compound']
    else:
        sentiment['sentiment_neutral'] = scores['compound']

    return sentiment


def get_entities_and_sentiment(reviews, airline):
    """
    Extracts the named entities and sentiment from the specified list of sentences.

    Parameters:
        sentences (list): A list of sentences to analyze.

    Returns:
        A list of dictionaries, where each dictionary contains the named entity counts and sentiment for a single sentence.
    """
    # Initialize the list of entity dictionaries and sentiment scores

    # Process each sentence
    temp_airline = []

    for row in reviews:

        verified = "Trip Verified" in row['review_text']
        sentence = row['review_text'].replace("Trip Verified", "")

        # Extract the named entities from the sentence
        entities = extract_entities(sentence)

        # Calculate the sentiment score for the sentence
        sentiment_score = get_sentiment(sentence)
        sentiment_score = {'sentiment':list(sentiment_score.keys())[0], 'sentiment_value':list(sentiment_score.values())[0]}

        # Create a dictionary containing the entity counts and sentiment for the sentence
        entity_dict = {
            'airline': airline,
            'verified': verified,
            'ORGANIZATION': 0,
            'PERSON': 0,
            'LOCATION': 0,
            'MONEY': 0,
            'DATE': 0,
            'TIME': 0,
        }
        # todo extract  entities 
        for entity in entities:
            entity_dict[entity[0]] = +1 

        entity_dict.update(sentiment_score)
        entity_dict.update(row)
        temp_airline.append(entity_dict)
    # Return a list of dictionaries containing the entity counts and sentiment for each sentence
    return temp_airline


def crawl(url,airline,true):
    response = requests.get(url)

    if response.status_code == 200:
        try:

            soup = BeautifulSoup(response.content, "html.parser")
            reviews = []
            articles = soup.select(".comp_media-review-rated")
            for article in articles:
                try:
                    # Extract review ratings
                    ratings = {}
                    rating_elements = soup.find_all('td', class_='review-rating-header')

                    for element in rating_elements:
                        rating_name = element.text.strip()
                        rating_value_element = element.find_next_sibling('td').find_all('span', class_='star fill')
                        rating_value = len(rating_value_element)
                        ratings[rating_name] = rating_value

                    # Extract recommended value
                    recommended_element = soup.find('td', class_='review-rating-header recommended')
                    recommended_value = recommended_element.find_next_sibling('td').text.strip()
                    recommended = recommended_value.lower() == 'yes'
                    # Store the extracted information in a dictionary

                    title = article.select_one(".text_header").text.strip()
                    author = article.select_one("[itemprop='name']").text.strip()
                    rating = article.find('span', itemprop='ratingValue').text.strip()
                    published_time = article.select_one("[itemprop='datePublished']").get("content")
                    review_text = article.select_one(".text_content").text.strip()

                    review_text = article.select_one(".text_content").text.strip()

                    review_info = {}
                    review_info["review_title"] = title
                    review_info["review_author"] = author
                    review_info["review_rating"] = rating
                    review_info["published_time"] = datetime.strptime(published_time, "%Y-%m-%d").date()
                    review_info["MonthYear"] = published_time
                    review_info["review_text"] = review_text
                    review_info.update( {"Rating Value": rating_value,
                        **ratings,
                        "Recommended": recommended})

                    headers = ['Type Of Traveller', 'Seat Type', 'Route', 'Date Flown']

                    # Iterate over each header and print its value
                    for header in headers:
                        try:
                            row = article.find('td', text=header).find_next_sibling()
                            value = row.text
                            review_info[header] = value
                        except:
                            review_info[header] = None


                    reviews.append(review_info)

                except Exception as e:
                    print ("Exception For Loop")
                    print (e)
            res = get_entities_and_sentiment(reviews,airline)
            return res
        except Exception as e:
            print ("exception")
            print (e)
            print (url)

            return None
    else:
        return None


error = []
for airline_elem in urls:
    url = airline_elem['url']
    airline = airline_elem['name']
    url_pagination = airline_elem['url_pagination']
    temp_def = crawl(url,airline,True)
    if temp_def:
        result.extend(temp_def)
    else:
        error.append(url)

    pagination = True
    enumeration = 2

df = pd.DataFrame.from_dict(result)
df.to_csv('export_airlines_complete.csv',index=False)
print (error)