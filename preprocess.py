# preprocess.py
import pandas as pd
from transformers import BertTokenizer

class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def load_data(self):
        data = pd.read_csv(self.filepath)
        questions = data['question'].tolist()
        answers = data['answer'].tolist()
        return questions, answers

    def tokenize_data(self, texts, max_length=128):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
