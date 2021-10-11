import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Loads the specified datasets and merges to one dataframe
    Parameters
    ----------
    messages_filepath : str
        Path to csv with messages
    categories_filepath: str
        Path to csv with categories associated with messages
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame for the two datasets
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(categories, messages, on="id")
    return df


def clean_data(df):
    """ Cleans the dataset to atomic columns, numeric datatypes, removing duplicates
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with merged Datasets
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    categories = df["categories"].str.split(";", expand=True)
    categories["id"] = df["id"]
    
    row = categories.iloc[0,:]
    category_colnames = row.str.replace("-[0-9]", "", regex=True)
    category_colnames[-1] = "id"
    categories.columns = category_colnames
    
    for column in categories:
        if column != "id":
            categories[column] = categories[column].str.strip().str[-1]
            categories[column] = pd.to_numeric(categories[column])
    
    df = df.drop("categories", axis=1)
    df = pd.merge(df, categories, on="id")
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """ Saves the data to a sqlite database in the 'disaster_tweets' table
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be inserted
    database_filename: str
        Path to sqlite database

    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_tweets', engine, index=False)


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