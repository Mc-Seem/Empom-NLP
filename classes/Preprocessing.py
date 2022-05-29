import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer, PorterStemmer
from pymorphy2 import MorphAnalyzer


def df_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw chat-bot dataframe into a more interpretable one, specifically:
     - removes redundant fields
     - explodes in chat lines
     - parses the sender and marks whether it is user, bot, operator or comment
     - renames the fields

    Parameters:
        df: (pd.DataFrame): an initial dataframe

    Returns:
        df: (pd.DataFrame): a transformed dataframe


    """

    to_drop = ['Дата начала', 'Длительность чата', 'Тип чата', 'Название канала', 'Данные пользователя', 'Первый вопрос(ы)', 'Закрыт',
               '% участия бота', '% участия рекомендаций', '% участия оператора', 'Варианты ответов бота', 'Оценка чата', 'Название оценки',
               'Комментарий оценки', 'Почта операторов', 'Время на первый ответ, сек', 'Переменные чата']
    df = df.drop(to_drop, axis=1)
    df['user'] = df['Пользователь'].str.replace(r'\n-', '')
    df['user'] = df['user'].str.replace(r'\n.*', '')
    df['operators'] = df['Операторы'].str.replace(r'\n', '')
    df['operators'] = df['operators'].apply(lambda x: {} if type(x) is float else set(x.split(',')))
    df['chat_lines'] = df['Содержание чата'].str.findall(r'\d\d:\d\d:\d\d .*')

    df = df.explode('chat_lines')

    df['time'] = df['chat_lines'].str.slice(stop=8)
    df['sender'] = df['chat_lines'].apply(lambda x: x[9:9 + x[9:].find(':')]) # here the logic is a little harder
    df['sender'] = df['sender'].str.replace(r' \(рекомендация\)', '')
    df['sender_name'] = df['sender']
    df['line'] = df['chat_lines'].apply(lambda x: x[9 + x[9:].find(':') + 2:])

    df = df[df['line'] != '']
    df = df[df['user'] != '']

    df = df.drop(['Пользователь', 'Содержание чата', 'Операторы', 'chat_lines'], axis=1)
    df = df.rename(columns={'ID чата': 'chat_id', 'Тип канала': 'channel_type', 'Тематики': 'topics', 'Документы': 'documents',
                            'Реакция на ответы бота': 'reaction', 'Уверенность бота': 'bot_confidence', 'Среднее время на ответ, сек': 'mean_response_time'})

    df.loc[df['sender'] == df['user'], 'sender'] = 'user'
    df.loc[df['sender'] == 'Бот', 'sender'] = 'bot'
    df.loc[df['sender'] == 'Комментарий', 'sender'] = 'comment'

    operator_map = []
    for index, r in df.iterrows():
        operator_map.append(r['sender'] in r['operators'])
    df.loc[operator_map, 'sender'] = 'operator'

    df = df.drop('operators', axis=1)

    return df


class EM_Pomoshnik_TextPreprocessor:
    """
    Class that provides particular text preprocessing for EM.Pomoshnik (for more info check transform())
    """

    def __init__(self, replacing_order: str = '', replacing_incident: str = '', replacing_shop: str = ''):
        """
        Initiates an instance of a EM.Pomoshnik text preprocessor

        Parameters:
            replacing_order: (str): word that will replace the order codes (by default is an empty string)
            replacing_incident: (str): word that will replace the incident codes (by default is an empty string)
            replacing_shop: (str): word that will replace the shop codes (by default is an empty string)
        """
        self.replacing_order = replacing_order
        self.replacing_incident = replacing_incident
        self.replacing_shop = replacing_shop

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
          - order codes
          - incident codes
          - shop codes

        Returns:
            self.data: (pd.Series): Transformed data
        """
        self.data = self.data.str.lower()

        self.__delete_order_codes()
        self.__delete_incident_codes()
        self.__delete_shop_codes()
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

    def __delete_order_codes(self) -> None:
        """
        Removes order codes (e.g. 1526872280)
        """
        self.data = self.data.str.replace(r'\d\d\d\d\d\d\d\d\d\d', self.replacing_order)
        self.data = self.data.str.replace(r'\d\d\d\d\d\d\d\d\d', self.replacing_order)

    def __delete_shop_codes(self) -> None:
        """
        Removes shop codes (e.g. Sa25)
        """
        self.data = self.data.str.replace(r'[a-zA-Z][a-zA-Z]\d\d', self.replacing_shop)

    def __delete_incident_codes(self) -> None:
        """
        Removes incident codes (e.g. 21-17929533)
        """
        self.data = self.data.str.replace(r'\d\d-\d\d\d\d\d\d\d\d', self.replacing_incident)

    def save_order_codes(self) -> pd.DataFrame:
        """
        Extracts all order codes from the data

        Returns:
            Dataframe of all order codes in the data corresponding to the original part of data where it was found
        """
        temp = self.data.copy()
        temp.index = temp
        return temp.str.extractall(r'(\d\d\d\d\d\d\d\d\d\d)')

    def save_incident_codes(self) -> pd.DataFrame:
        """
        Extracts all incident codes from the data

        Returns:
            Dataframe of all incident codes in the data corresponding to the original part of data where it was found
        """
        temp = self.data.copy()
        temp.index = temp
        return temp.str.extractall(r'(\d\d-\d\d\d\d\d\d\d\d)')


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

        self.__delete_links()
        self.__delete_punctuation()
        self.__delete_numbers()
        self.__change_to_ye()
        self.__delete_stop_words()

        if self.method == 'stem' or self.method == 'stemming':
            self.__stem()
        elif self.method == 'lemma' or self.method == 'lemmatize' or self.method == 'lemmatization':
            self.__delete_latin()  # otherwise russian lemmatize will break down
            self.__lemmatize()

        self.__delete_short()  # не
        self.__delete_stop_words()
        self.__delete_whitespace()

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
        self.data = self.data.str.replace(r'[^\w\s]', ' ')

    def __delete_short(self) -> None:
        """
        Removes words of 2 characters or less
        """
        self.data = self.data.str.replace(r'\s[A-zА-я]{1,2}\s', ' ')

    def __delete_numbers(self) -> None:
        """
        Removes numbers
        """
        self.data = self.data.str.replace(r'\d+', self.replacing_word)

    def __change_to_ye(self) -> None:
        """
        Changes ё to e
        """
        self.data = self.data.str.replace('ё', 'е')

    def __delete_links(self) -> None:
        """
        Removes all URLs
        """
        self.data = self.data.str.replace(r'http\S+', ' ')
        self.data = self.data.str.replace(r'www\S+', ' ')

    def __delete_latin(self) -> None:
        """
        Removes all words written with latin alphabet
        """
        self.data = self.data.str.replace(r'[A-Za-z\s]', ' ')

    def __delete_whitespace(self) -> None:
        """
        Removes all whitespace characters
        """
        self.data = self.data.str.replace(r'\s+', ' ')
