# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 02-09-2021
# =============================================================================
"""
This module implements helper classes to extract events. The problem 
    is formalized as a zero-shot text classification task. All 
    models in this class apply cross-encoder architecture; therefore, 
    though high performance, it can be very slow with too many classes.
"""

# %%
import os
from pathlib import Path
from typing import List, Union, Any, Dict

import numpy as np
from transformers import pipeline


class BARTEventExtractor:
    
    def __init__(self, model: str = 'all-MiniLM-L6-v2', events: List[str] = []):

        self._events = tuple(events)
        self._model = pipeline('zero-shot-classification', model)
        self._top_n_events = 4


class SentBERTEventExtractor:
    
    def __init__(self, model: str = 'cross-encoder/nli-MiniLM2-L6-H768', events: List[str] = []):

        self._events = tuple(events)
        self._model = pipeline('zero-shot-classification', model)
        self._top_n_events = 4
