import re


class TextProcessing(object):

    def __init__(self, text):
        self.text = text

    def remove_stopwords(self, stop_words, tokens):
        res = []
        for token in tokens:
            if not token in stop_words:
                res.append(token)
        return res

    def clean_phrase(self):
        text = self.text
        text = text.encode('ascii', errors='ignore').decode()
        text = text.lower()
        text = re.sub('\W|\s', ' ', text)
        text = re.sub(r'\s+[a-zA-Z]\s+|\d|http\S+', " ", text)
        text = text.strip()
        return text

