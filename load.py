import pandas as pd
import csv


def load_GloVe():
    print("Loading GloVe 50D...")
    glove_file = "/home/jaf/PycharmProjects/e-NABLE-Classifier/data/GloVe/glove.6B.50d.txt"
    GloVe = pd.read_table(glove_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    print("Load Complete!")
    return GloVe