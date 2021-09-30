from unittest import TestCase

import torch
from transformers import AutoTokenizer

from wisdomify.datasets import WisdomDataModule
from wisdomify.vocab import VOCAB


class TestWisdomDataModule(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_module = WisdomDataModule(
            data_version='latest',
            data_name='init_wisdom2eg-raw',
            dtype='example',
            k=11,
            device='cpu',
            vocab=VOCAB,
            tokenizer=AutoTokenizer.from_pretrained("beomi/kcbert-base"),
            batch_size=10,
            num_workers=1,
            shuffle=False,
            repeat=1,
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

