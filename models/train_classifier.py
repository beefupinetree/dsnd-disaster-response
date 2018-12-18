import sys
import re
import nltk
import pickle
import warnings
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
warnings.filterwarnings("ignore")


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine).loc[:99]
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


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf',
         MultiOutputClassifier(RandomForestClassifier(random_state=42),
                               n_jobs=-1))
    ])

#    pipeline = Pipeline([
#        ('vect', CountVectorizer(tokenizer=tokenize)),
#        ('tfidf', TfidfTransformer()),
#        ('clf',
#         MultiOutputClassifier(RandomForestClassifier(random_state=42)))
#    ])
#    parameters = {'clf__estimator__estimator__kernel': ['rbf', 'linear',
#                                                        'poly'],
#                  'clf__estimator__estimator__C': [0.5, 1, 2]
#                  }
#    parameters = {'clf__estimator__kernel': ['rbf']
#                  }
    parameters = {'clf__estimator__n_estimators': [10, 20],
                  'clf__estimator__min_samples_split': [2, 4]
                  }
#    cv = GridSearchCV(pipeline, param_grid=parameters,
#                      scoring='precision_samples', cv=5)
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


#def evaluate_model(model, X_test, Y_test, category_names):
#    y_pred = model.predict(X_test)
#
#    print('Best score and parameter combination = ')
#    print(model.best_score_)
#    print(model.best_params_)
#    for i, column in enumerate(Y_test.columns):
#        print("{}. {}:".format(i+1, column))
#        print(classification_report(Y_test[Y_test.columns[i]], y_pred[:,i],
#                                    target_names=category_names))

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    print('---------------------------------')
    for i in range(Y_test.shape[1]):
        print('{0:>25s} accuracy : {1:.2f}'.format(category_names[i],
              accuracy_score(Y_test[:, i],
                             Y_pred[:, i])))


def save_model(model, model_filepath):
    filename = model_filepath + "\\optimal_model.pkl"
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
