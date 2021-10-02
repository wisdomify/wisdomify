from flask import Flask, jsonify, request, render_template_string
from wisdomify.loaders import load_device
from wisdomify.wisdomifier import Wisdomifier


class WisdomifierAPI:
    def __init__(self, ver: str):
        device = load_device()
        self.wisdomifier = Wisdomifier.from_pretrained(ver, device)
        print(f'wisdomifier loaded -> ver: {ver}')


app = Flask(__name__)
wisdomifier_0 = WisdomifierAPI(ver="0")
wisdomifier_1 = WisdomifierAPI(ver="1")


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
        
        <a href="/gwageo"><button>GwaGeo</button></a>
        <a href="/search"><button>Search</button></a>
        <a href="/api"><button>API</button></a>
    </body>
    </html>
    """


@app.route('/search', methods=['GET'])
def wisdomifySearch():
    desc = request.args.get('desc')
    ver = str(request.args.get('ver'))

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
            
            <form method="GET" action="/search" >
                <div>
                    <select name="ver">
                        <option value="None">Select Version</option>
                        <option value="0">0</option>
                        <option value="1">1</option>
                    </select>
                </div>
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
                    <h4> Version: {{ ver }}</h4>
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
        ver=ver,
        desc=desc,
        desc_result=globals()[f'wisdomifier_{ver}'].wisdomifier.wisdomify(sents=[desc]) if desc else None
    )


@app.route('/api', methods=['GET'])
def wisdomifyAPI():
    desc = request.args.get('desc')
    ver = str(request.args.get('ver'))
    if desc:
        return jsonify(list(map(
            lambda results: dict(map(
                lambda res:
                (res[0], res[1]),
                results
            )),
            globals()[f'wisdomifier_{ver}'].wisdomifier.wisdomify(sents=[desc])
        )))

    return jsonify(None)


@app.route('/gwageo', methods=['GET'])
def GwaGeo():
    desc = request.args.get('desc')

    return render_template_string(
        """
        <html>
        <head>
            <title>
                wisdomify
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

            <form method="GET" action="/gwageo" >
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
                        <th>Version_0</th>
                        <th>Version_0</th>
                        <th></th>
                        <th>Version_1</th>
                        <th>Version_1</th>
                        
                      </tr>
                      <tr>
                        <th>속담</th>
                        <th>확률</th>
                        <th></th>
                        <th>속담</th>
                        <th>확률</th>
                      </tr>

                    {% for results in desc_result %}
                        {% for res in results %}
                        <tr>
                            <td>{{ res[0][0] }}</td>
                            <td>{{ res[0][1] }}</td>
                            <td></td>
                            <td>{{ res[1][0] }}</td>
                            <td>{{ res[1][1] }}</td>
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
        desc_result=[list(zip(
            wisdomifier_0.wisdomifier(sents=[desc])[0],
            wisdomifier_1.wisdomifier(sents=[desc])[0],
        ))] if desc else None
    )


@app.route('/healthz', methods=['GET'])
def checkHealth():
    return "Alive", 200


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0', threaded=True)

