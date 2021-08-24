from abc import ABC
import json
import logging
import os
import zipfile

import torch
import transformers

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM, AutoConfig,
)
from ts.torch_handler.base_handler import BaseHandler

with zipfile.ZipFile("wisdomify.zip", 'r') as zip_ref:
    zip_ref.extractall('./')

from wisdomify.builders import build_vocab2subwords
from wisdomify.models import RD, Wisdomifier
from wisdomify.vocab import VOCAB

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class WisdomifierHandler(BaseHandler, ABC):
    """
    wisdomifier handler
    """

    def __init__(self):
        super(WisdomifierHandler, self).__init__()
        self.initialized = False
        self.setup_config = dict()
        self.wisdomifier = None

    def initialize(self, ctx):
        """
        model loaded.
        wisdomifier initialised.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties

        # serialised file
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        # system device check (CPU, CUDA)
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        # read configs for the mode, model_name, etc. from build_setting.json
        setup_config_path = os.path.join(model_dir, "build_setting.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the build_setting.json file.")

        bert_model: str = self.setup_config['bert_model']
        k: int = self.setup_config['max_length']

        bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert_model))
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        vocab2subwords = build_vocab2subwords(tokenizer, k, VOCAB).to(self.device)
        rd = RD.load_from_checkpoint(model_pt_path, bert_mlm=bert_mlm, vocab2subwords=vocab2subwords)
        rd.eval()  # otherwise, the model will output different results with the same inputs
        rd.to(self.device)

        self.wisdomifier = Wisdomifier(rd, tokenizer)

        logger.info(
            "wisdomifier loaded successfully", model_dir
        )

        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing.
        This stage, nothings are preprocessed.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            str : preprocessed text for query
        """

        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode('utf-8')

            logger.info("Received text: '%s'", input_text)

        return input_text

    def inference(self, input_text, **kwargs):
        """Predict the class (or classes) of the received text.
        Args:
            input_text: query text, input sentence of the model.
        Returns:
            list(dict) : It returns a list(dict) of inference result.
        """

        return list(map(
            lambda results: dict(map(
                lambda res:
                (res[0], res[1]),
                results
            )),
            self.wisdomifier.wisdomify(sents=[input_text])
        ))

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list(dict)): It contains the predicted response of the input text.
        Returns:
            (list(dict)): Returns a list(dict) of the Predictions and Explanations.
        """
        return inference_output
