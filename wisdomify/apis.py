import argparse
import re

from flask import jsonify, request
from flask_classful import FlaskView
from flask_cors import cross_origin

from wisdomify import flows
from wisdomify.connectors import connect_to_wandb, connect_to_es

from wisdomify.docs import Story
from wisdomify.flows import SearchFlow, Wisdom2DefFlow
from wisdomify.loaders import load_config
from wisdomify.wisdomifier import Wisdomifier


class WisdomifyView(FlaskView):
    """
    sents -> wisdoms
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default='a',
                        required=True,
                        help="This parameter is used for wisdomifyAPI deployment."
                             "The parameter should be the model version described on WandB.")
    parser.add_argument("--model", type=str,
                        default='rd_beta',
                        required=True,
                        help="This parameter is used for wisdomifyAPI deployment."
                             "The parameter should be the model name described on WandB.")

    args = parser.parse_args()
    ver = args.ver
    model = args.model

    config = load_config()[model][ver]
    with connect_to_wandb(job_type="deploy", config=config) as run:
        # --- init a wisdomifier --- #
        flow = flows.ExperimentFlow(run, model, ver)("d", config)
        flow.rd_flow.rd.eval()
    # --- wisdomifier is independent of wandb run  --- #
    wisdomifier = Wisdomifier(flow.rd_flow.rd, flow.datamodule)

    @cross_origin(origin='*')
    def index(self):
        """
        Wisdomify endpoint returning a list of wisdoms along with scores from prediction.
        ---
        parameters:
          - name: sent
            in: query
            type: string
            required: true
          - name: size
            in: query
            type: string
            required: true
        responses:
          200:
            description: A list of two list. Inner list contains either wisdoms or scores in descending order of score.
            schema:
              type: array
              items:
                type: array
                items:
                  type: string
              example: [
                  [
                    "갈수록 태산",
                    "산 넘어 산",
                    "꿩 대신 닭",
                    "원숭이도 나무에서 떨어진다",
                    "가는 날이 장날"
                  ],
                  [
                    0.35413333773612976,
                    0.32364851236343384,
                    0.2984081208705902,
                    0.016462184488773346,
                    0.004309742245823145
                  ]
                ]

        """
        form = request.args
        sent = form['sent']
        size = int(form['size']) if 'size' in form.keys() else 10

        results = self.wisdomifier(sents=[sent])[0][:size]
        response = jsonify(
            list(zip(*sorted(results, key=lambda k: k[1], reverse=True)))
        )

        return response


class StorytellView(FlaskView):
    """
    wisdoms -> examples & definitions
    """
    es = connect_to_es()
    index_name = ",".join(name for name in Story.all_names())
    size = 5

    config = {'data': 'wisdoms', 'ver': 'a', 'val_ratio': None, 'seed': 410, 'upload': False}
    run = connect_to_wandb(job_type="download", config=config)
    wisdom2DefFlow = Wisdom2DefFlow(run, config['ver'])
    wisdom2DefFlow.download_raw_df()

    def egs(self):
        """
        Storyteller endpoint returning a list of dictionary containing index and highlighted sentence from ElasticSearch.
        ---
        parameters:
          - name: wisdom
            in: query
            type: string
            required: true
        responses:
          200:
            description: A list of dictionaries. The dictionary includes index name and highlighted sentence.
            schema:
              type: array
              items:
                type: object
                properties:
                  index:
                    type: string
                  highlight:
                    type: string
              example: [
                  {
                    "highlight": "수업 마무리 잘했어요?ㅎㅎ 웅~~ 버스 신세계! 나 벌써 도서관 도착 하지만.. 도착했어요?ㅎㅎ 벌써?? 월요일 휴관..? 도서관이라니 헐...? 아이고ㅜㅜ ㅋㅋㅋㅋ 가는날이장날 가는날이 장날.. ㅋㅋㅋ 통했당ㅋㅋ 그르니까",
                    "index": "ksns_story"
                  },
                  {
                    "highlight": "#@이름#!!!!!일본가나!!!!!! 개부럽다!!!!!!! 부럽 ㅠ 일본 도착~! 삿포로 비온다 ㅎ #@시스템#사진# 호엑 비옴? 가는날이 장날 ㅠ 와 여기 겁나 춥다 ㅎ",
                    "index": "ksns_story"
                  },
                  {
                    "highlight": "책 빌렸낭! 오늘야심차게갔더니 야심차게 정기휴관일 딱! 써있더라 이런이런 가는날이 장날이라고 내일 빌려야지",
                    "index": "ksns_story"
                  },
                  {
                    "highlight": "비와 ㅇㅅㅇ #@시스템#사진# 엥? 비와? 헐 소나기는 여기가 아니고 거기였구나 소나기인듯 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 가는날이 장날이네 완전 여긴 여전히 해가 부글부글 끓고있음",
                    "index": "ksns_story"
                  },
                  {
                    "highlight": "짱난당 시립도서관왓드니 ㅋㅋㅋㅋ 휴관일이네?:) 돈애낄라고 커피 사들고왓는데ㅠㅠ #@이모티콘# ㅋㅋㅋㅋㅋㅋㅋㅋ기름값쓰 아깝네여ㅋㅋㅋㅋㅋ 가는날이장날이네요 ㅋㅋ웃겨ㅋㅋㅋ",
                    "index": "ksns_story"
                  }
                ]
          400:
            description: When no response from ElasticSearch.
        """
        form = request.args
        wisdom = form['wisdom']

        flow = SearchFlow(self.es, self.index_name, self.size)(wisdom)
        res = flow.res

        if res:
            return jsonify([
                {
                    "index": hit['_index'],
                    "highlight": re.sub('[<em></em>]', '', hit['highlight']['sents'][0])
                }
                for hit in res['hits']['hits']
            ])

        else:
            return jsonify(f"No data for '{wisdom}'"), 404

    def defs(self):
        """
        Storyteller endpoint returning a list of definition of the wisdom.
        ---
        parameters:
          - name: wisdom
            in: query
            type: string
            required: true
        responses:
          200:
            description: A list of definitions of the wisdom
            schema:
              type: array
              items:
                type: string
              example: [
                  "어떤 일을 하려고 하는데 뜻하지 않은 일을 공교롭게 당함",
                  " 어떤 일을 마음먹고 하려는데, 뜻밖의 문제로 일이 계획대로 되지 않고 곤란을 겪음",
                  "우연히 갔다가 예상치 못한 일이 벌어졌음",
                  "계획해 왔던 일을 할 때 생각하지 않았던 상황이 벌어짐",
                  "계획해 왔던 일을 할 때 예상치 못한 일이 벌어짐"
                ]
          400:
            description: When no wisdom in wisdom2def table.
        """
        form = request.args
        wisdom = form['wisdom']

        defs = self.wisdom2DefFlow.raw_df\
            .loc[self.wisdom2DefFlow.raw_df['wisdom'] == wisdom]['def']\
            .to_list()

        if defs:
            return jsonify(defs)

        else:
            return jsonify(f"No data for '{wisdom}'"), 404

