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


def loadData():
    """
    Takes a csv file and outputs a pandas dataframe.
    :param csvpath:
    :return:
    """
    csvpath = "./data/csv/akshai_labels.csv"
    columns = ["content", "Report", "Device", "Delivery", "Progress",
             "becoming_member", "attempt_action" , "Activity", "Other"]

    eNABLE_df = pd.read_csv(csvpath, usecols=columns, keep_default_na=False)
    return eNABLE_df


def __rowToVec(row):
    outputVector = ([row["Report"]], [row["Device"]], [row["Delivery"]], [row["Progress"]], [row["becoming_member"]],
                    [row["attempt_action"]], [row["Activity"]], [row["Other"]])
    return outputVector


def getTrainTest():
    """
    Generates files train and test csv
    :param testRate: Percentage of dataframe to be set aside for test data
    :return:
    """
    df = loadData()
    nlp = spacy.load('en')
    max_encoder_time = 500
    df["y_term"] = df.apply(__rowToVec, axis=1).apply(np.array)

    df["x_term"] = df["content"].apply(lambda c: [token.text.lower() for token in nlp(c)][:500])
    # df = df[df["x_term"].map(len) < 1000] # Remove vectors which have higher dimentions
    df["x_term"].apply(lambda c: pad(c, max_encoder_time))

    # Split dataframe as a random sample with 80% train to test ratio
    train = df.sample(frac=0.8, random_state=200)
    test = df.drop(train.index)

    return train, test


def pad(c, max_encoder_time):
    shape = max_encoder_time-len(c)
    c.extend(shape * ["_PAD"])

