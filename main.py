from pickle import TRUE
import string
from nltk import word_tokenize, SnowballStemmer, ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from dataset_generator import load_or_extract_dataset
from models import cross_val_results, testing, training, results
from sklearn.model_selection import train_test_split


def get_ngrams(tokens, num):
    """
    Returns a list of unigram/bigram/trigram/n-gram combinations

    :param tokens:
    :param num:
    :return:
    """
    return [' '.join(grams) for grams in ngrams(tokens, num)]


def text_preprocessing(text):
    """
    Applies basic text preprocessing to get a clean set of tokens
    Converts all to lowercase, splits text into words, removes noise and stems words

    :param text:
    :return: Processed tokens
    """
    # Step 1: Convert all letters to lowercase
    text = text.lower()

    # Step 2: Split text into tokens (words)
    tokens = word_tokenize(text)

    # Step 3: Remove noise (stopwords, punctuation, numbers)
    text_stopwords = stopwords.words("english")

    # Add to stopwords some words that are found in the set, but missed from nltk
    text_stopwords.extend(['--', '---', '..', "''", '...', '``'])
    tokens = [token for token in tokens if
            not token.isdigit() and not token in string.punctuation and token not in text_stopwords]
    
    # Step 4: Stem words (e.g. charging and charged should not be treated differently)
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(token) for token in tokens]

    # Step 5: Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]

    return tokens


def uni_big_vector(vectorizer, train_data, test_data):
    if vectorizer == 'unigram':
        vectorizer = CountVectorizer(min_df=2, max_df = 0.7)
    elif vectorizer == "bigram":
        vectorizer = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=2, max_df = 0.7)
    X = vectorizer.fit_transform(train_data)
    Y = vectorizer.transform(test_data)

    transformer = TfidfTransformer( smooth_idf=False)
    train = transformer.fit_transform(X).toarray()
    test = transformer.fit_transform(Y).toarray()

    return train, test


def get_results_to_csv(algorithm, accuracy, precision, recall, f1, ngram, parameters):
    
    df_results = pd.DataFrame(columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'n-gram', 'Parameters'])
    temp_list = []

    # model things
    classifier = re.compile("(.*?)\s*\(").match(str(algorithm.best_estimator_)).group(1)
    temp_list.append(classifier)
    temp_list.append(accuracy)
    temp_list.append(precision)
    temp_list.append(recall)
    temp_list.append(f1)
    temp_list.append(ngram)
    temp_list.append(parameters)

    df_results.loc[len(df_results)] = temp_list
    df_results.to_csv('results_4.csv', mode='a')
    

    

def main():
    # Load dataset
    dataset = load_or_extract_dataset()

    # Apply text preprocessing to each text
    dataset['tokens'] = dataset['text'].apply(text_preprocessing)
    dataset['cleantext'] = dataset['tokens'].apply(lambda x: ' '.join(x))
 
    # Optional: save_dataset(dataset, 'processed_dataset.pkl')
    train_data = dataset[dataset["fold"] != "fold5"]
    test_data = dataset[dataset["fold"] == "fold5"]


    x_train_data = pd.Series(train_data.iloc[:, 1].values)
    y_train_data = pd.Series(train_data.iloc[:, 3].values).astype('int')
    x_test_data = pd.Series(test_data.iloc[:, 1].values)
    y_test_data = pd.Series(test_data.iloc[:, 3].values).astype('int')

    #unigram and bigram 
    df_train_uni, df_test_uni = uni_big_vector('unigram', x_train_data, x_test_data)
    df_train_bi, df_test_bi = uni_big_vector('bigram', x_train_data, x_test_data)
    #split the training data
    x_train_uni, x_test_uni, y_train_uni, y_test_uni = train_test_split(df_train_uni, y_train_data, test_size=0.2, random_state= 40)
    x_train_bi, x_test_bi, y_train_bi, y_test_bi = train_test_split(df_train_bi, y_train_data, test_size=0.2, random_state= 40)
    #training and cross validation
    nb_model_uni, lr_model_uni, dt_model_uni, rf_model_uni = training(x_train_uni, y_train_uni, x_test_uni)
    nb_model_bi, lr_model_bi, dt_model_bi, rf_model_bi = training(x_train_bi, y_train_bi, x_test_bi)

    models_uni = [nb_model_uni, lr_model_uni, dt_model_uni, rf_model_uni]
    models_bi = [nb_model_bi, lr_model_bi, dt_model_bi, rf_model_bi]               

    for model in models_uni:
        #print the best parameter's values
        #parameters = cross_val_results(model)
        #testing the data
        accuracy, precision, recall, f1, conf_matrix, class_report = testing(model, df_test_uni, y_test_data)
        #print the results
        results(model, accuracy, precision, recall, f1, conf_matrix, class_report)
        

    for model in models_bi:
        #print the best parameter's values
        #parameters = cross_val_results(model)
        #testing the data
        accuracy, precision, recall, f1, conf_matrix, class_report = testing(model, df_test_bi, y_test_data)
        #print the results
        results(model, accuracy, precision, recall, f1, conf_matrix, class_report, True)
        
    return dataset

main()
