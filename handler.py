from abc import ABC
import json
import logging
import os
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import io

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        # Read model serialize/pt file
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir) 
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        # Read the mapping file, index to object name
        # mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        # if os.path.isfile(mapping_file_path):
        #     with open(mapping_file_path) as f:
        #         self.mapping = json.load(f)
        # else:
        #     logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')
        logger.info("model and processor loading done")  
        self.initialized = True

    def preprocess(self, data):        
        # img = cv2.imread(data)
        logger.info("preprocess started")
        print(data)
        img = np.asarray(Image.open(io.BytesIO(data[0]["body"])))
        inputs=(cv2.resize(img,(384,384)))
        inputs=cv2.cvtColor(inputs,cv2.COLOR_GRAY2RGB)
        logger.info("input image shape")
        logger.info(inputs.shape)
        logger.info("preprocessing  done")
        return inputs

    def inference_custom(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        # NOTE: This makes the assumption that your model expects text to be tokenized  
        # with "input_ids" and "token_type_ids" - which is true for some popular transformer models, e.g. bert.
        # If your transformer model expects different tokenization, adapt this code to suit 
        # its expected input format.
        logger.info("inference started") 
        pixel_values = self.processor(images=inputs, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(output)
        data=[]
        data.append(output)
        print("data output")
        print(data)
        return data

    # def postprocess_custom(self, inference_output):
    #     # TODO: Add any needed post-processing of the model predictions here
    #     return inference_output


_service = TransformersClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference_custom(data)
        # data = _service.postprocess_custom(data)

        return data
    except Exception as e:
        raise e
