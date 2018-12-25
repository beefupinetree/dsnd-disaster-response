import sys
import re
import nltk
import pickle
import warnings
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
warnings.filterwarnings("ignore")


def load_data(database_filepath):
    '''
    input:
        database_filepath: File path of the database created from
        process_Data.py.
    output:
        X: 1-d df with all the messages.
        Y: df of the categories assigned to each message.
        category_names: List of names of the possible categories.
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message'].values
    # the 'genre', 'original', and 'id' variables were dropped
    Y = df.loc[:, 'related':].values
    category_names = df.columns[4:]
    return X, Y, category_names


def tokenize(text):
    '''
    input:
        text: Message data for tokenization.
    output:
        clean_tokens: Resulting list after tokenization.
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    input:
        N/A
    output:
        cv: model pipeline and grid-search parameters.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',
         MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    parameters = {'clf__estimator__n_estimators': [10, 20],
                  'clf__estimator__min_samples_split': [2, 4]
                  }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    input:
        model: Fitted classification model.
        X_test: df of messages to test our model
        Y_test: df of the actual categories
        category_names: List of names of the categories to be classified.
    output:
        N/A: Prints a classification report to the console.
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    print()
    for i in range(Y_test.shape[1]):
        print('{0:>25s} accuracy : {1:.2f}'.format(category_names[i],
              accuracy_score(Y_test[:, i],
                             Y_pred[:, i])))


def save_model(model, model_filepath):
    '''
    input:
        model: Trained classification model.
        model_filepath: Path where the model will be saved.
    output:
        N/A: saves model to designated path.
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
