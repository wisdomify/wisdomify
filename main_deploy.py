import argparse

from flask import Flask

from wisdomify.apis import WisdomifyView, StorytellView

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


if __name__ == '__main__':
    check_params()

    StorytellView.register(app)
    WisdomifyView.register(app)
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=False)

