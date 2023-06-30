# import required module
import os
import pandas as pd
import pickle


def extract_dataset(data_folder='data'):
    """
    Reads all text files located in the data directory and generates a dataset as a Pandas Dataframe

    :return: Pandas Dataframe
    """

    # Initialize columns
    columns = ['hotel_name', 'text', 'polarity', 'label']

    # Initialize options
    polarities = ['negative']
    labels = ['deceptive', 'truthful']

    # Initialize dataset
    dataframe = pd.DataFrame(columns=columns)

    # assign directory
    project_directory_name = os.path.dirname(__file__)
    data_directory = os.path.join(project_directory_name, data_folder)

    # Iterate each polarity
    for polarity in polarities:
        folder = data_directory + '/' + polarity + '_polarity'

        # Iterate each folder in polarity
        for label in labels:
            label_folder = folder + '/' + label

            # Iterate each fold
            for fold in range(1, 6):
                fold_folder = label_folder + '/fold' + str(fold)

                # iterate over files in that directory
                for filename in os.listdir(fold_folder):
                    file_path = os.path.join(fold_folder, filename)

                    # Get hotel name
                    hotel_name = filename.split('_')[1]
                    text = get_text(file_path)

                    # Add record to dataset
                    if label == 'deceptive':
                        dataframe = append_row(dataframe, {'hotel_name': hotel_name, 'text': text, 'polarity': polarity,
                                                        'fold': "fold" + str(fold), 'label': 0})
                    else:
                        dataframe = append_row(dataframe, {'hotel_name': hotel_name, 'text': text, 'polarity': polarity,
                                                        'fold': "fold" + str(fold), 'label': 1})
    return dataframe


def get_text(text_file_path):
    """
    Returns the text from the given text file

    :param text_file_path:
    :return:
    """
    with open(text_file_path, 'r') as file:
        return file.read().replace('\n', '')


def append_row(df, record_as_dict):
    """
    Appends a new row to the given Pandas dataframe

    :param df: Dataframe (Pandas)
    :param record_as_dict: Record with data in form of dictionary
    :return: A dataframe with the new record appended
    """

    df = pd.concat([df, pd.DataFrame.from_records([record_as_dict])])
    return df


def save_dataset(dataframes, pickle_file_name='/dataset.pkl'):
    """
    Saves the given dataset for future use

    :return: void
    """

    dataframes.to_pickle(os.path.dirname(__file__) + pickle_file_name)


def load_dataset(pickle_file_name='/dataset.pkl'):
    """
    Loads and returns the dataset from a pickle file

    :return: Pandas Dataframe
    """

    return pd.read_pickle(os.path.dirname(__file__) + pickle_file_name)


def load_or_extract_dataset(pickle_file_name='/dataset.pkl'):
    """
    Loads and returns the dataset from a pickle file iff the file exists
    If the file does not exist then the function will generate a new dataset
    and save it as pickle file

    :return: Pandas Dataframe
    """
    try:
        return load_dataset()
    except:
        dataframe = extract_dataset()
        pd.to_pickle(dataframe, os.path.dirname(__file__) + pickle_file_name)
        return dataframe
