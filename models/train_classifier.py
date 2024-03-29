import sys
import pandas as pd 
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer
from sqlalchemy import create_engine 
import re 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from  sklearn.metrics import classification_report
import pickle 


def load_data(database_filepath):
    '''
    This function loads data and stores in a sqlite database. 
    Input: database_filepath 
    Output: X-messages from the disaster site 
            Y-onehotencodered categories 
            labels - category names (labels)
    '''
    db_url = f'sqlite:///{database_filepath}'
    engine = create_engine(db_url)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.loc[:, 'related': 'direct_report']
    labels = Y.columns
    return X, Y, labels

def tokenize(text):
    '''
    Text normalisation, lemmatisation, tokenisation. 
    Input: Text - this is a string. 
    Output: Token_clean - normalised and lemmatised tokens 
    '''
    wnl = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    text_lower = text.lower()
    text_norm = re.sub(r'[^a-zA-Z0-9]', ' ', text_lower)
    tokens = word_tokenize(text_norm)
    tokens_clean = [wnl.lemmatize(word,pos = 'v') for word in tokens if word not in stop_words]
    tokens_clean = [wnl.lemmatize(word, pos = 'n') for word in tokens_clean]
    return tokens_clean 

def build_model(X_train, Y_train):
    '''
    Returns a classification multioutput pipeline, that includes a vectoriser, 
    a tfidftransformer, and randomforest classifiers. The model parameters are 
    determined via Cross-Validation Grid Search.
    ouput: model pipeline 
    '''
    pipeline = Pipeline(
        [('vectoriser', CountVectorizer(tokenizer= tokenize)),
        ('transformer', TfidfTransformer()), 
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))]
    )
    parameters = {
        'classifier__estimator__n_estimators':[50, 100], 
        'classifier__estimator__criterion':['entropy'], 
        'classifier__estimator__min_samples_leaf': [10, 20]
    }

    cv = GridSearchCV(pipeline, parameters, scoring= 'f1_weighted', refit=True)
    cv.fit(X_train, Y_train)
    best_pipe = cv.best_estimator_
    return best_pipe 


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Prints a classification report for the multioutput model 
    Input: Model - train model pipeline 
            X_test, Y_test - test data set including features, and target variable , respectively 
            category_names - a vector of the same number of columns as Y_test containing the category names 
            save_model 
    '''    
    y_test_predict = model.predict(X_test)
    for idx, class_column in enumerate (category_names):
        y_true = Y_test[class_column]
        y_predict = y_test_predict[:, idx]
        print(f'\n label: {class_column}\n')
        print(classification_report(y_true, y_predict))


def save_model(model, model_filepath):
    '''
    saves the model objects to a pickle file
    Inputs: Model object
            Model_filepath 
            
    '''
    with open(model_filepath, 'wb') as file: 
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model: run cross-validation grid search and fit best model...')
        model = build_model(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()