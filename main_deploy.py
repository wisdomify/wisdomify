import argparse

from flask import Flask
from flask import jsonify, request

from wisdomify.apis import WisdomifierAPI, StorytellerAPI

wisdomifier_api = WisdomifierAPI()
storyteller_api = StorytellerAPI()

app = Flask(__name__)


def check_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default=None,
                        help="This parameter is used for wisdomifyAPI deployment."
                             "The parameter should be the model version described on WandB.")

    args = parser.parse_args()
    ver = args.ver

    if ver is None:
        raise ValueError("'--ver' should be stated for 'wisdomify' deployment.")


@app.route('/wisdomify', methods=['GET'])
def sents2wisdoms():
    """
    sents -> wisdoms
    """
    form = request.json
    sent = form['sent']

    return jsonify(wisdomifier_api.infer(sent))


@app.route('/storytell', methods=['GET'])
def wisdoms2sents():
    """
    wisdoms -> sents
    """
    form = request.json
    sent = form['wisdom']

    return jsonify(storyteller_api.infer(sent))


@app.route('/healthz', methods=['GET'])
def checkHealth():
    return "Alive", 200


if __name__ == '__main__':
    check_params()

    app.run(host='0.0.0.0', port=8080, debug=False)
