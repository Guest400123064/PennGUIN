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
from abc import ABC, abstractmethod
from pprint import pprint

import re
import json
import string
import unicodedata

import numpy as np

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
        pass
    
    def predict_occupation(self, ):
        pass
    
    
    