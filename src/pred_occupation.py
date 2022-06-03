# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 05-29-2022
# =============================================================================
"""
This module implements helper classes to predict occupation of a person given a
    short text description of that entity. We formulate such a problem as a 
    zero-shot-text-classification problem and relies on Huggingface pipeline 
    for implementation (relies on natural language inference task). One good 
    theoretical foundation of NLI for zero-shot-classification can be found at
    <https://arxiv.org/abs/1909.00161>
"""

# %%
from typing import List, Union, Any, Dict
from pprint import pprint

import re
import string
import unicodedata

import torch
from transformers import pipeline
from datasets import Dataset


class HuggingfaceZerShotOccupationPredictor:
    
    def __init__(self, model: str = 'valhalla/distilbart-mnli-12-1', batch_size: int = 32):
        """
        Args:
            model (str, optional): pretrained-model name. Defaults to 'valhalla/distilbart-mnli-12-1'.
            batch_size (int, optional): Defaults to 32.
        """

        self._model_card = model
        self._model = pipeline(
            'zero-shot-classification', model, 
            device=(0 if torch.cuda.is_available() else -1)  # Using the first GPU only
        )
        self._batch_size = batch_size
        
    def __repr__(self):
        return f'HuggingfaceZerShotOccupationPredictor({self.model_card}, {self.batch_size})'
    
    @property
    def model_card(self):
        return self._model_card
    
    @property
    def model(self):
        return self._model
    
    @property
    def device(self):
        return self.model.device
    
    @property
    def batch_size(self):
        return self._batch_size
    
    def preprocess(self, s: str) -> str:
        """String pre-processing function, used to reduce noise.
            1. Convert all characters to ASCII
            2. Remove other irrelevant stuff like email address or external url
            3. Remove special symbols like newline character \\n"""
            
        # Normalize special chars
        s = (unicodedata.normalize('NFKD', s)
                .encode('ascii', 'ignore').decode())

        # Remove irrelevant info
        s = re.sub(r'\S*@\S*\s?', '', s)     # Email
        s = re.sub(r'\S*https?:\S*', '', s)  # URL
        
        # Keep punctuation and words only
        pattern_keep = (string.punctuation + 
                            string.ascii_letters + 
                            string.digits + 
                            r' ')
        return re.sub(r'[^' + pattern_keep + r']+', '', s)
    
    def get_occupation(self, texts: Union[str, List[str]], occupations: List[str]) -> List[Dict[str, Any]]:
        """Implementation using Huggingface ZeroShotClassification pipeline as the backend"""

        if isinstance(texts, str):
            texts = self.preprocess(texts)
            return self._get_batch([texts], occupations)
        elif isinstance(texts, list):
            texts = [self.preprocess(t) for t in texts]
            return self._get_batch(texts, occupations)
        else:
            raise ValueError('@ HuggingfaceZerShotOccupationPredictor.predict() :: ' + 
                f'Invalid <texts> type {type(texts)}; only <str, List[str]> allowed')
    
    def _get_batch(self, texts: List[str], occupations: List[str]) -> List[Dict[str, Any]]:
        """Prediction driver function"""
        
        # Helper function for batched inference
        def _predict(batch, pipe, labels):
            return {'outputs': pipe(batch['texts'], candidate_labels=labels)}
        
        # Create a huggingface dataset for batched processing
        outputs = (Dataset
            .from_dict({'texts': texts})
            .map(_predict, batched=True, batch_size=self.batch_size, 
                    fn_kwargs={'labels': occupations, 'pipe': self.model})
            .to_dict()
            .get('outputs'))
        
        # Standardize output dict keys
        for i, o in enumerate(outputs):
            outputs[i] = {
                'text': o['sequence'],
                'occupations': o['labels'],
                'scores': o['scores']
            }
        return outputs


# Sample usage
if __name__ == '__main__':
    
    sample_texts = [
        "Jacob Gedleyihlekisa Zuma is a South African politician who was the fourth president of South Africa from 2009 to 2018. He is also referred to by his...",
        "Thomas Matthew Bradby (born 13 January 1967) is a British journalist and novelist who currently presents the ITV News at Ten. He was previously political...",
        "Eddie Jerome Vedder is an American singer, musician, and songwriter best known as the lead vocalist and one of four guitarists of the rock band Pearl Jam."
    ]
    sample_occupations = [
        'politician',
        'businessperson',
        'journalist',
        'social activist',
        'extremist',
        'judge',
        'lawyer',
        'economist',
        'critic',
        'military personnel',
        'artist'
    ]
    
    # Predict occupation
    predictor = HuggingfaceZerShotOccupationPredictor()
    pprint(predictor.get_occupation(sample_texts, sample_occupations))
    
# %%
