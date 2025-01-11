import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function to load the .csv files
    Args:
        messages_filepath - str of filepath to messages.csv
        categories_filepath - str of filepath to categories.csv
    Returns:
        df: dataframe of merged input from the two .csv
    '''
    # load datasets
    messages = pd.read_csv('messages.csv')
    categories = pd.read_csv('categories.csv')

    # merge datasets
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    '''
    Function to clean the df
    Args:
        df - dataframe of merged input from the two .csv
    Returns:
        df: cleaned dataframe ready for saving in database
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # applying a lamda function to subselect a str disregarding the last 2 chars
    # of the first row in categories. This generates a list of new column names
    category_colnames = categories.iloc[0].apply(lambda x: x[0:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    '''
    Function saves the dataframe
    Args:
        df - dataframe of merged input from the two .csv
        database_filename - str of name of the db file
    Returns:
        - 
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Responses', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        # print(df.head())
        
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