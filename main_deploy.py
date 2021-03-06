import argparse

from flask import Flask
from flask_cors import CORS, cross_origin
from flasgger import Swagger

from wisdomify.apis import WisdomifyView, StorytellView

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, resources={r"/*": {"origin": "*"}})

app.config['SWAGGER'] = {
    "swagger_version": "2.0",
    "title": "Leaf",
    "version": "0.0.1",
    "headers": [

    ],
}

swagger = Swagger(app)


@cross_origin(origin='*')
@app.route('/healthz', methods=['GET'])
def checkHealth():
    return "Alive", 200


if __name__ == '__main__':
    StorytellView.register(app)
    WisdomifyView.register(app)
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=False)
