import pandas as pd
import numpy as np 
from sqlalchemy import create_engine 
import sys 
import argparse 


def load_data(messages_filepath, categories_filepath):
    '''
    Load messages and categories from input file path. 
    Input: 
    messages_filepath - location of messages file 
    categories_filepath - locaction of categories file 
    Output: 
    df - dataframe countaining the merged messages and categories 
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df= messages.merge(categories, on= 'id', how= 'inner')
    return df


def clean_data(df):
    '''
    clean the dataset countained in df by converting categories to binary variables, 
    adding appropriate column names and removing duplicates.
    Input: 
    df - a dataframe produced by load_data()
    Output: 
    df - the clean dataset 
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat= ';', n= -1, expand= True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    # category_colnames = row.str.split(pat='-', n=  1, expand= True)[0] # This implementation uses pandas function to exttract the column names. 
    category_colnames = row.apply(lambda x: x[0:len(x)-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        categories[column] = np.where(categories[column] >1, 1, categories[column])
    
    # drop the original categories column from `df`
    df= df.drop('categories', axis= 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis =1)
    
    df = df.drop_duplicates()
    
    return(df)

def save_data(df, database_filename):
    '''
    saves the dataframe df to a sqlite database indicated in the input 
    Input: 
    df - pandas dataframe, 
    database_filename - location of sqlite database 
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False, if_exists = 'replace')

def main(messages_filepath, categories_filepath, database_filepath):
    '''
    run the process data procedures : load, clean and saves data 
    Input: 
    messages_filepath - location of messages file 
    categories_filepath - locaction of categories file 
    database_filename - location of sqlite database
    '''
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
            .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)


    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser('process Data for Disaster Response')
    parser.add_argument('-messages_file', help= 'path to messages file.')
    parser.add_argument('-categories_file', help= 'path to categories file.')
    parser.add_argument('-db_file', help= 'path to database file')
    args = parser.parse_args()
    main(args.messages_file, args.categories_file, args.db_file)
    