"""
    Preprocesses the e-NABLE facebook posts using GloVe word Embeddings

    author: Japheth Adhavan
"""
import pandas as pd
import numpy as np
import spacy


def loadGloVe():
    """
    Loads the GloVe embeddings to memory for easy access.

    Makes use of spacey's GloVe binaries.

    Usage:
        python -m spacy download en_vectors_web_lg
    :return: SpaCy object
    """
    print("Loading GloVe vectors to memory...")
    nlp = spacy.load('en_vectors_web_lg')
    print("GloVe Data Loaded!")
    return nlp


def loadData(csvpath):
    """
    Takes a csv file and outputs a pandas dataframe.
    :param csvpath:
    :return:
    """
    columns = ["content", "Report", "Device", "Delivery", "Progress",
             "becoming_member", "attempt_action" , "Activity", "Other"]

    eNABLE_df = pd.read_csv(csvpath, usecols=columns, keep_default_na=False)
    return eNABLE_df


def __rowToVec(row):
    outputVector = [
                    row["Report"],
                    row["Device"],
                    row["Delivery"],
                    row["Progress"],
                    row["becoming_member"],
                    row["attempt_action"],
                    row["Activity"],
                    row["Other"]
                    ]
    return outputVector


def getTrainTest():
    """
    Generates files train and test csv
    :param testRate: Percentage of dataframe to be set aside for test data
    :return:
    """
    df = loadData()
    nlp = loadGloVe()

    df["y_term"] = df.apply(__rowToVec, axis=1)
    df["x_term"] = df["content"].apply(lambda c: [[token.vector] for token in nlp(c)])

    # Split dataframe as a random sample with 80% train to test ratio
    train = df.sample(frac=0.8, random_state=200)
    test = df.drop(train.index)

    return train, test


def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[validate_end:]]
    return train, validate, test


def save_data_splits():
    csvpath = "./data/raw_data/akshai_labels.csv"
    eNABLE_df = loadData(csvpath)
    train, validate, test = train_validate_test_split(eNABLE_df)
    train.to_csv("./data/split_data/train.csv", index=False)
    validate.to_csv("./data/split_data/validate.csv", index=False)
    test.to_csv("./data/split_data/test.csv", index=False)




