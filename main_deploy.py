import argparse

from flask import Flask

from wisdomify.apis import WisdomifyView, StorytellView

app = Flask(__name__)


@app.route('/healthz', methods=['GET'])
def checkHealth():
    return "Alive", 200


if __name__ == '__main__':

    StorytellView.register(app)
    WisdomifyView.register(app)
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=False)

