from typing import List
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer, PorterStemmer
from pymorphy2 import MorphAnalyzer


class TextPreprocessor:
    """
    Class that provides basic text preprocessing (more detailed info in transform())
    and extracts all numbers from the data (more info in save_numbers())
    """

    def __init__(self, method: str, stop_words: list = [], pos: bool = False, replacing_word: str = ' '):
        """
        Initiates an instance of a text preprocessor

        Parameters:
            method: (str): Name of a preprocessing method (either stemming or lemmatization)
            stop_words: (list): List of stop-words to be removed from the data
            pos: (bool): Whether to apply pos-tagging during lemmatization (if chosen)
            replacing_word: (str): Word with which the numbers in the data will be replaced (by default an empty string)
        """
        if not method in {'lemmatization', 'lemmatize', 'lemma', 'stemming', 'stem'}:
            raise Exception("This preprocessing method is not supported")

        self.method = method
        self.data = pd.Series(dtype=object)
        self.pos = pos
        self.replacing_word = replacing_word

        if self.method == 'stem' or self.method == 'stemming':
            ru_stemmer = SnowballStemmer(language='russian')
            en_stemmer = PorterStemmer()
            self.stop_words = [en_stemmer.stem(ru_stemmer.stem(x.lower())) for x in stop_words]
        elif self.method == 'lemma' or self.method == 'lemmatize' or self.method == 'lemmatization':
            morph = MorphAnalyzer()
            self.stop_words = [morph.normal_forms(x.lower())[0] for x in stop_words]

    def fit(self, data: pd.Series) -> None:
        """
        Fits the data into the preprocessor

        Parameters:
            data: (pd.Series): Data to be fit
        """
        self.data = data

    def transform(self) -> pd.Series:
        """
        A function that removes:
            - stop-words
            - punctuation
            - words of 2 characters and less
            - numbers
            - links
            - whitespace characters
        Changes ё to е
        Performs the selected method of preprocessing:
            - stemming
            - lemmatization

        Returns:
            self.data: (pd.Series): Transformed data
        """
        self.data = self.data.str.lower()

        self.__delete_punctuation()
        self.__delete_whitespace()
        self.__delete_numbers()
        self.__delete_links()
        self.__change_to_ye()


        if self.method == 'stem' or self.method == 'stemming':
            self.__stem()
        elif self.method == 'lemma' or self.method == 'lemmatize' or self.method == 'lemmatization':
            # self.__delete_latin()  # otherwise russian lemmatize will break down
            self.__lemmatize()

        self.__delete_stop_words()
        self.__delete_short()

        return self.data

    def fit_transform(self, data: pd.Series) -> pd.Series:
        """
        Fits, then transforms data (check fit() and transform() for deeper info)

        Parameters:
            data: (pd.Series): Data to be fit
        Returns:
            self.data: (pd.Series): Transformed data
        """
        self.fit(data)
        return self.transform()

    def __stem(self) -> None:
        """
        Applies russian and english stemming using nltk's Snowball and Porter stemming methods
        """
        ru_stemmer = SnowballStemmer(language='russian')
        en_stemmer = PorterStemmer()
        self.data = self.data.apply(lambda x: ' '.join([en_stemmer.stem(ru_stemmer.stem(y)) for y in x.split()]))

    def __lemmatize(self) -> None:
        """
        Applies pymorhpy2 lemmatization and pos-tagging (if chosen in __init__)
        """
        morph = MorphAnalyzer()
        if self.pos:
            self.data = self.data.apply(
                lambda x: ' '.join([(morph.normal_forms(i)[0] + '_' + morph.parse(i)[0].tag.POS) for i in x.split()]))
        else:
            self.data = self.data.apply(lambda x: ' '.join([morph.normal_forms(i)[0] for i in x.split()]))

    def __delete_stop_words(self) -> None:
        """
        Removes stop-words provided in __init__
        """
        pattern = r'\b(?:{})\b'.format('|'.join(self.stop_words))
        self.data = self.data.str.replace(pattern, '')

    def __delete_punctuation(self) -> None:
        """
        Removes punctuation
        """
        self.data = self.data.str.replace(r'[^\w\s]', '')

    def __delete_short(self) -> None:
        """
        Removes words of 2 characters or less
        """
        self.data = self.data.str.replace(r'[ ]+..[ ]+|[ ]+.[ ]+', ' ')

    def __delete_numbers(self) -> None:
        """
        Removes numbers
        """
        self.data = self.data.str.replace(r'\b\d+\b', self.replacing_word)

    def __change_to_ye(self) -> None:
        """
        Changes ё to e
        """
        self.data = self.data.str.replace('ё', 'е')

    def __delete_links(self) -> None:
        """
        Removes all URLs
        """
        self.data = self.data.str.replace(r'http\S+', '')

    def __delete_latin(self) -> None:
        """
        Removes all words written with latin alphabet
        """
        self.data = self.data.str.replace(r'[A-Za-z\s]', '')

    def save_numbers(self) -> List:
        """
        Extracts all numbers from the data

        Returns a list of all numbers in the data
        """
        return self.data.str.extractall(r'\b\d+\b')[0].tolist()

    def __delete_whitespace(self) -> None:
        """
        Removes all whitespace characters
        """
        self.data = self.data.str.replace(r'\s+', ' ')
