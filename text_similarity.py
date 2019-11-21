import numpy as np
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
import tensorflow_hub as hub
import nltk
from nltk.corpus import stopwords
from nlp_test_project.text_processing import TextProcessing
nltk.download('stopwords')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"


class TextSimilarity(object):

    def __init__(self):
        pass

    def _get_features(self, sentence):
        if type(sentence) is str:
            sentence = [sentence]
            print('get_features ', sentence)
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            embed = hub.Module(module_url)
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            return sess.run(embed(sentence))

    def _process_text(self, sent):
        sent = TextProcessing(sent).clean_phrase()
        return ' '.join(TextProcessing(sent).remove_stopwords(stopwords.words('english'), sent.split()))

    def _cosine_similarity(self, vec1, vec2):
        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)
        return np.dot(vec1, vec2) / (norm_v1 * norm_v2)

    def test_similarity(self, sentence1, sentence2):
        vec1 = self._get_features(self._process_text(sentence1))[0]
        vec2 = self._get_features(self._process_text(sentence2))[0]
        return self._cosine_similarity(vec1, vec2)

    def check_similarity(self, sentence1, sentence2):
        sim_percent = self.test_similarity(sentence1, sentence2)
        print(sim_percent)
        if sim_percent > 0.50:
            return "similar topics " + str(sim_percent)
        else:
            return "not similar topics " + str(sim_percent)

