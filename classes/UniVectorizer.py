import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class Vectorizer:
    """
    Unified vectorizer class to vectorize text
    """
    def __init__(self, v_type: str, **params):
        """
        Initiates an instance of a chosen vectorizer with given key-word parameters

        Parameters:
            v_type: (str): vectorizer type - (either tfidf, count, that stand for
            TfidfVectorizer, CountVectorizer from sklearn respectively)

            **params: (key-word args): parameters to be passed
        """

        self.__vectorizer = None
        self.__data = None
        if not v_type in {'tfidf', 'count'}:
            raise Exception('This vectorizer is not supported')

        if v_type == 'tfidf':
            self.__vectorizer = TfidfVectorizer()
        elif v_type == 'count':
            self.__vectorizer = CountVectorizer()

        for p_key in params.keys():
            if not p_key in self.__vectorizer.get_params().keys():
                raise Exception(f'There is no such parameter in {type(self.__vectorizer)}')
        self.__vectorizer.set_params(**params)

    def fit(self, X: pd.Series) -> None:
        """
        Fits data into vectorizer

        Parameters:
            X: (pd.Series): data to be fit
        """
        self.__data = X
        self.__vectorizer.fit(X)

    def get_feature_names(self) -> list:
        """
        Returns a list of feature names (only applicable to tfidf and count)

        Returns:
            features: (list): feature names
        """
        return self.__vectorizer.get_feature_names()

    def get_stop_words(self) -> list:
        """
        Builds or fetches the effective stop words list

        Returns:
            stop_words: (list): stop words
        """
        return self.__vectorizer.get_stop_words()

    def transform(self, X: pd.Series=None) -> pd.DataFrame:
        """
        Transforms a sequence of documents to a document-term matrix. Uses vocabulary learned by fit.

        Parameters:
            X: (pd.Series): sequence to be transformed

        Returns:
            data: (pd.DataFrame): transformed data
        """
        if X is None:
            return self.__vectorizer.transform(self.__data)
        return self.__vectorizer.transform(X)

    def fit_transform(self, X: pd.Series, y=None) -> pd.DataFrame:
        """
        First fits, then transforms given data (for further info read fit() and transform()

        Parameters:
            X: (pd.Series): data to be fit and transformed

        Returns:
            data: (pd.DataFrame): transformed data

        """
        return self.__vectorizer.fit_transform(X)

    def get_params(self, deep=True) -> dict:
        """
        Returns parameters, currently used in the vectorizer

        Returns:
             params: (dict): dictionary with vectorizer parameters
        """
        return self.__vectorizer.get_params(deep)

    def set_params(self, **params) -> None:
        """
        Sets vectorizer's parameters to chosen params

        Parameters:
            **params: (key-words args): parameters to be passed
        """
        self.__vectorizer.set_params(**params)

    def build_analyzer(self) -> callable:
        """
        Builds and returns a callable function that preprocesses input data and generates tokens and
        n-grams based on it

        Returns:
            analyzer: (callable): built analyzer function
        """
        return self.__vectorizer.build_analyzer()

    def build_tokenizer(self) -> callable:
        """
        Builds and returns a callable function for splitting a string into a sequence of tokens

        Returns:
            tokenizer: (callable): built tokenizer function
        """
        return self.__vectorizer.build_tokenizer()

    def build_preprocessor(self) -> callable:
        """
        Builds and returns a callable function for preprocessing text before tokenization

        Returns:
            preprocessor: (callable): built preprocessing function
        """
        return self.__vectorizer.build_preprocessor()

    def decode(self, doc: str) -> str:
        """
        Decodes input into a utf-8 string

        Parameters:
            doc: (str): input to be decoded

        Returns:
            doc: (str): decoded string
        """
        return self.__vectorizer.decode(doc)
