import torch
import yaml
from flask import Flask, jsonify, request, render_template_string
from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer
from wisdomify.builders import build_vocab2subwords
from wisdomify.loaders import load_conf
from wisdomify.models import RD, Wisdomifier
from wisdomify.paths import WISDOMIFIER_HPARAMS_YAML, WISDOMIFIER_CKPT
from wisdomify.vocab import VOCAB


class WisdomifierAPI:
    def __init__(self, ver: int):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        conf = load_conf()
        vers = conf['versions']
        if ver not in vers.keys():
            raise NotImplementedError("Invalid version provided".format(ver))

        selected_ver = vers[ver]
        bert_model: str = selected_ver['bert_model']

        wisdomifier_path = WISDOMIFIER_CKPT.format(ver=ver)
        with open(WISDOMIFIER_HPARAMS_YAML.format(ver=ver), 'r') as fh:
            wisdomifier_hparams = yaml.safe_load(fh)
        k = wisdomifier_hparams['k']

        bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert_model))
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        print('bert_mlm and tokenizer loaded')
        vocab2subwords = build_vocab2subwords(tokenizer, k, VOCAB).to(device)
        rd = RD.load_from_checkpoint(wisdomifier_path, bert_mlm=bert_mlm, vocab2subwords=vocab2subwords)
        rd.eval()  # otherwise, the model will output different results with the same inputs
        rd = rd.to(device)  # otherwise, you can not run the inference process on GPUs.
        wisdomifier = Wisdomifier(rd, tokenizer)

        self.wisdomifier = wisdomifier
        print('wisdomifier loaded')


app = Flask(__name__)
wisdomifierAPI = WisdomifierAPI(ver=0)


@app.route('/', methods=['GET'])
def wisdomifyHome():
    return """
    <html>
    <head>
        <title>
            wisdomify - version_0
        </title>
        <style>
        </style>
    </head>
    <body>
        <h1>wisdomify</h1><br>
        
        <a href="/search"><button>Search</button></a>
        <a href="/api"><button>API</button></a>
    </body>
    </html>
    """


@app.route('/search', methods=['GET'])
def wisdomifySearch():
    desc = request.args.get('desc')
    if desc:
        desc_result = wisdomifierAPI.wisdomifier.wisdomify(sents=[desc])
        return render_template_string(
            """
            <html>
            <head>
                <title>
                    wisdomify - version_0
                </title>
                <style>
                table, th, td {
                  border: 1px solid black;
                  margin: 1px;
                  padding: 1px;
                }
                </style>
            </head>
            <body>
                <h1>wisdomify</h1><br>
                <form method="GET" action="/search">
                    <div>
                        <label for="desc"> 검색할 문장을 입력하세요 </label>
                        <input type="text" name="desc">
                    </div>
                    <div class='button'>
                        <button type="submit">검색하기</button>
                    </div>
                </form>
                <p>
                    {% if desc == None %}
                        <h5> 문장을 입력해주세요. </h5>
                    {% else %}
                        <h3> 검색어: {{ desc }}</h3>
                        <table style="width:100%">
                          <tr>
                            <th>속담</th>
                            <th>확률</th>
                          </tr>
                        
                        {% for results in desc_result %}
                            {% for res in results %}
                            <tr>
                                <td>{{ res[0] }}</td>
                                <td>{{ res[1] }}</td>
                            </tr>
                            {% endfor %}
                        </table>
                        {% endfor %}
                    {% endif %}
                </p>
            </body>
            </html>
            """,
            desc=desc,
            desc_result=desc_result
        )
    return render_template_string(
        """
            <html>
            <head>
                <title>
                    wisdomify - version_0
                </title>
                <style>
                </style>
            </head>
            <body>
                <h1>wisdomify</h1><br>
                <form method="GET" action="/search">
                    <div>
                        <label for="desc"> 검색할 문장을 입력하세요 </label>
                        <input type="text" name="desc">
                    </div>
                    <div class='button'>
                        <button type="submit">검색하기</button>
                    </div>
                </form>
            </body>
            </html>
            """
    )


@app.route('/api', methods=['GET'])
def wisdomifyAPI():
    desc = request.args.get('desc')
    if desc:
        return jsonify(list(map(
            lambda results: dict(map(
                lambda res:
                (res[0], res[1]),
                results
            )),
            wisdomifierAPI.wisdomifier.wisdomify(sents=[desc])
        )))

    return jsonify(None)


@app.route('/healthz', methods=['GET'])
def checkHealth():
    return "Alive", 200


if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0', threaded=True)
