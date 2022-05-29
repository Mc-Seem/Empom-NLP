from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm.notebook import tqdm

label_dct = {0: 'negative', 1: 'neutral', 2: 'positive'}

tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")

model_path = "auxiliary/labse_sentiment_finetuned/"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)


def tokenize_function_sentences(examples):
    """
    Auxiliary function to simplify sentence tokenization
    """
    return tokenizer(examples, padding=True, truncation=True, max_length=300, return_tensors='pt')


def get_sentiments(texts, batch_size=10):
    """
    Function to get the sentiments for a list of texts

    Parameters:
        texts: (list(str)): list of texts to be sentimentally assessed
        batch_size: (int): batch size to tweak RAM load and swiftness of the process

    Returns:
        predictions: (list(int)): list of predicted sentiment values, where 0 - negative, 1 - neutral, 2 - positive
    """

    predictions = []
    for i in tqdm(range(0, len(texts), batch_size)):
        X_tokens = tokenize_function_sentences(texts[i:i + batch_size])
        test_preds = model(**X_tokens).logits.detach().numpy()
        test_preds = [np.argmax(i) for i in test_preds]
        predictions += test_preds

    for p in predictions:
        p = label_dct[p]

    return predictions
