import nltk #for natural language processing 
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd

#nltk pakages 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#sklearn pakages

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(messages_filepath, categories_filepath):
    engine = create_engine('sqlite:///etl_df.db')
df =pd.read_sql_table('etl_df',engine) #change the df names for processing
X = df['message']
y= df.iloc[:, 4:] #selecting rows and columns for features 
print(X.head(5))
    pass


def clean_data(df):
    df = pd.merge(messages,categories)
df.head()
categories = df["categories"].str.split(";", n=36, expand=True)
categories.head()
row = categories.iloc[0]
category_colnames = row.apply(lambda x: x[:-2])
print(category_colnames)
categories.columns = category_colnames
categories.head()
for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1]  
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])
categories.head()

df.drop('categories',axis = 1, inplace = True)
df.head(7)
frames = [df,categories]
df = pd.concat(frames, axis=1)
df.head()
df["is_duplicate"]= df.duplicated()
print(df.head())
df.drop_duplicates(subset='id', inplace=True)
df["is_duplicate"]= df.duplicated()
print(df.tail())
    pass


def save_data(df, database_filename):
    from sqlalchemy import create_engine

engine = create_engine('sqlite:///etl_df.db')
df.to_sql('etl_df', engine, index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()