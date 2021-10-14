import unittest
from unittest import TestCase
from transformers import AutoTokenizer

import torch

from wisdomify.experiment import Experiment
from wisdomify.utils import WandBSupport


@unittest.skip("이 테스트는 로컬에서 실행시켜주세요. "
               "WandB 로그인 권한이 필요하기 때문에 Github Action에서는 skip됩니다."
               "로컬에서 실행시 이 데코레이터를 주석처리 해주시고, git push시 원상복구 해주세요!")
class TestWisdomDataModule(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        config = {
            'data_version': 'latest',
            'data_name': 'init_wisdom2eg-raw',
            'data_type': 'example',
            'k': '11',
            'wisdoms': [
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
            ],
            'batch_size': 10,
            'num_workers': 1,
            'shuffle': False,
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
