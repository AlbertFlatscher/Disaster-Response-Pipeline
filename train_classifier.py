import sys

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

# import libraries
import numpy as np
import pandas as pd
import pickle
import statistics
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import FeatureUnion
from custom_transformer import StartingVerbExtractor

# disable warnings
import warnings
warnings.filterwarnings("ignore")

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table('Responses', engine)
    X = df.message.values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values

    category_names = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns

    return X, Y, category_names


def tokenize(text):
    # tokenize text
    word_list = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # iterate through each token of word list
    clean_tokens = []
    for tok in word_list:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # build pipeline conaining a CountVectorizer, StartingVerbExtractor and a
    # MultiOutputClassifier using AdaBoost as an estimator
    Ada_SVE_pipeline = Pipeline([
        ('features', FeatureUnion([

                    ('text_pipeline', Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer())
                    ])),

                    ('starting_verb', StartingVerbExtractor())
                ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators=100, learning_rate=0.01)))
    ])

    return Ada_SVE_pipeline


def evaluate_model(model, X_test, y_test, category_names):
    '''
    Function to determine model performance scores
    Args:
        model - model to be investigated
        X_test - testing data for model input
        y_test - testing data for model input
        category_name - list of Classes to predict
    Returns:
        y_pred: prediction of the model
    '''
    # make predictions
    y_pred = model.predict(X_test)

    # calculate scores for each Class
    reports_list = []
    precisions_list = []
    recalls_list = []
    f1_sores_list = []

    for i in range(y_test.shape[1]):
        reports_list.append(classification_report(y_test[:, i], y_pred[:, i]))
        precisions_list.append(precision_score(y_test[:, i], y_pred[:, i], average='weighted'))
        recalls_list.append(recall_score(y_test[:, i], y_pred[:, i], average='weighted'))
        f1_sores_list.append(f1_score(y_test[:, i], y_pred[:, i], average='weighted'))

    # print best Parameters
    try:
        print("\nBest Parameters:", model.best_params_)
    except:
        print("\nBest Parameters cant be determined since no GridSearch was used")

    # print mean accuracy over all Classes
    accuracy = (y_pred == y_test).mean()
    print("Accuracy:", accuracy)

    # print print mean precision over all Classes
    precision = statistics.mean(precisions_list)
    print("Precision:", precision)

    # print mean recall over all Classes
    recall = statistics.mean(recalls_list)
    print("Recall:", recall)

    # print mean F1 score over all Classes
    try:
        f1 = statistics.mean(f1_sores_list)
        print("F1-score:", f1)
    except Exception as e:
        print("F1 score could not be determined")
        print(e)


def save_model(model, model_filepath):
    # Saving the trained model as a pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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