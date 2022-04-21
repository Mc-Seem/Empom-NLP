import numpy as np
import pandas as pd


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
    df = df.reset_index()

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
