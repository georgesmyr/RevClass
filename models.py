from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from mlxtend.evaluate import mcnemar_table
from statsmodels.stats.contingency_tables import mcnemar
from sklearn import metrics
import re

def results(model, accuracy, precision, recall, f1_score, conf_matrix, class_report, Bigram=False):
    """
    Print the results of the methods

    :param algorithm:
    :param accuracy:
    :param precision:
    :param recall:
    :param f1_score:
    :param conf_matrix:
    :return:
    """

    # print the results 
    print(f"Best parameters were: {model.best_params_}")
    print("The accuracy of the test set :", accuracy)
    print("The precision of the test set :", precision)
    print("The recall of the test set :", recall)
    print("The F1-score of the test set :", f1_score)
    print("The classification report of the test set :\n", class_report)

    # set the Confusion Matrix plot
    confusion_matrix_plot = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    fig, ax = plt.subplots()
    confusion_matrix_plot.plot(ax=ax)
    confusion_matrix_plot.ax_.set(xlabel='Predicted', ylabel='True')
    confusion_matrix_plot.ax_.set_title('Confusion Matrix');
    if Bigram == True:
        plt.savefig('plots/confusion_matrix_' + str(re.compile("(.*?)\s*\(").match(str(model.best_estimator_)).group(1)) + '_UniBigram' + '.png')
    else:
        plt.savefig('plots/confusion_matrix_' + str(re.compile("(.*?)\s*\(").match(str(model.best_estimator_)).group(1)) + '_Unigram'  + '.png')


def cross_val_results(model):
    return model.best_params_


def training(x_train, y_train, x_test):
    """
    Training and cross validation of the algorithms with different parameters

    :param x_train:
    :param y_train:
    :return: models
    """

    nb_model = GridSearchCV(MultinomialNB(), {'alpha': [0.01, 0.1, 0.25, 0.5, 1, 5, 10]}, cv=4, n_jobs=-1)
    lr_model = GridSearchCV(LogisticRegression(solver='liblinear'), {'penalty': ['l1', 'l2'], 
                                                            'C': [0.01, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]}, cv=4, n_jobs=-1)
    dt_model = GridSearchCV(DecisionTreeClassifier(), {'max_depth': [None, 2, 4, 8, 16], 
                                                            'min_samples_split': [2, 4, 8, 16]}, cv=4, n_jobs=-1)
    rf_model = GridSearchCV(RandomForestClassifier(), {'max_depth': [None, 2, 4, 8, 16], 'min_samples_split': [2, 4, 8, 16],
                                                            'n_estimators': [10, 50, 100, 500]}, cv=4, n_jobs=-1)

    nb_model.fit(x_train, y_train)
    lr_model.fit(x_train, y_train)
    dt_model.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)

    nb_model.predict(x_test)
    lr_model.predict(x_test)
    dt_model.predict(x_test)
    rf_model.predict(x_test)

    return nb_model, lr_model, dt_model, rf_model

def predict(model, x_test):
    return model.predict(x_test)


def testing(model, x_test, y_test):
    """
    Testing the algorithms

    :param model:
    :param x_test:
    :param y_test:
    :return: metrics
    """

    accuracy = accuracy_score(y_test, predict(model, x_test))
    precision = precision_score(y_test, predict(model, x_test),labels=model.classes_, average='macro')
    recall = recall_score(y_test, predict(model, x_test),labels=model.classes_, average='macro')
    f1 = f1_score(y_test, predict(model, x_test),labels=model.classes_, average='macro')
    conf_matrix = confusion_matrix(y_test, predict(model, x_test), labels=model.classes_)
    class_report = metrics.classification_report(y_test, predict(model, x_test))

    return accuracy, precision, recall, f1, conf_matrix, class_report


