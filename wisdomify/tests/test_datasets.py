import unittest
from unittest import TestCase
from transformers import AutoTokenizer

import torch

from wisdomify.experiment import Experiment
from wisdomify.utils import WandBSupport


# @unittest.skip("이 테스트는 로컬에서 실행시켜주세요. "
#                "WandB 로그인 권한이 필요하기 때문에 Github Action에서는 skip됩니다."
#                "로컬에서 실행시 이 데코레이터를 주석처리 해주시고, git push시 원상복구 해주세요!")
class TestWisdomDataModule(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        config = {
            "exp_name": "version_0",
            "exp_desc": "[version_0]: The first minimal-viable-product of wisdomify. S_wisdom = S_wisdom_literal",
            "wandb": {
                "load": {
                    "rd_name": "rd_tiny_def_data",
                    "rd_ver": "",
                    "mlm_name": "mlm_init_kcbert",
                    "mlm_ver": "",
                    "tokenizer_name": "tokenizer_init_kcbert",
                    "tokenizer_ver": "",
                    "data_name": "init_wisdom2eg-raw",
                    "data_version": "",
                    "data_type": "example"
                },
                "save": {
                    "rd_name": "rd_tiny_def_data",
                    "rd_desc": "이 RD 모델은 50여개의 정의 데이터만을 사용해서 학습되었습니다.",
                    "mlm_name": "mlm_tiny_def_data",
                    "mlm_desc": "이 mlm 모델은 50여개의 정의 데이터만을 사용해서 학습되었습니다.",
                    "tokenizer_name": "",
                    "tokenizer_desc": ""
                }
            },
            "model": {
                "rd_model": "RDAlpha",
                "X_mode": "XBuilder",
                "y_mode": "YBuilder",
                "k": 11,
                "lr": 0.00001,
                "max_epochs": 40,
                "batch_size": 30,
                "num_workers": 0,
                "shuffle": True
            },
            "wisdoms": [
                "가는 날이 장날",
                "갈수록 태산",
                "꿩 대신 닭",
                "등잔 밑이 어둡다",
                "소문난 잔치에 먹을 것 없다",
                "핑계 없는 무덤 없다",
                "고래 싸움에 새우 등 터진다",
                "서당개 삼 년이면 풍월을 읊는다",
                "원숭이도 나무에서 떨어진다",
                "산 넘어 산"
            ]
        }
        device = torch.device('cpu')
        wandb_support = WandBSupport(ver='0', run_type='test_dataset', entity='artemisdicotiar')

        tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')

        cls.data_module = Experiment.build_datamodule(
            config=config,
            data_name='definition',
            tokenizer=tokenizer,
            k=11,
            device=device,
            wandb_support=wandb_support
        )

    def test_prepare_data(self):
        file_types = ['train', 'validation', 'test']
        self.data_module.prepare_data()
        self.assertEqual(list(self.data_module.story.keys()), file_types)

        self.assertIn('wisdom', self.data_module.story[file_types[0]][0].keys())
        self.assertIn('eg', self.data_module.story[file_types[0]][0].keys())

        self.assertIn('wisdom', self.data_module.story[file_types[1]][0].keys())
        self.assertIn('eg', self.data_module.story[file_types[1]][0].keys())

        self.assertIn('wisdom', self.data_module.story[file_types[2]][0].keys())
        self.assertIn('eg', self.data_module.story[file_types[2]][0].keys())

    def test_setup(self):
        self.data_module.prepare_data()
        self.data_module.setup()

        self.assertEqual(type(self.data_module.dataset_train.X), torch.Tensor)
        self.assertEqual(type(self.data_module.dataset_train.y), torch.Tensor)

        self.assertEqual(type(self.data_module.dataset_val.X), torch.Tensor)
        self.assertEqual(type(self.data_module.dataset_val.y), torch.Tensor)

        self.assertEqual(type(self.data_module.dataset_test.X), torch.Tensor)
        self.assertEqual(type(self.data_module.dataset_test.y), torch.Tensor)

    @classmethod
    def tearDown(cls):
        cls.data_module.wandb_support.push()
