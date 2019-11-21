#!flask/bin/python
from flask import Flask, request

from nlp_test_project.text_similarity import TextSimilarity

app = Flask(__name__)


@app.route('/post', methods=['GET', 'POST'])
def post():
    if request.method == 'POST':
        content = request.get_json(force=True)
        sentence1 = (content['sentence_1'])
        sentence2 = (content['sentence_2'])
        result = TextSimilarity().check_similarity(sentence1, sentence2)
        return result


if __name__ == '__main__':
    app.run(debug=True, port=5000)
