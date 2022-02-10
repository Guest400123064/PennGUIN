# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 02-09-2021
# =============================================================================
"""
This module implements helper classes to extract Goldstein events. The problem 
    is formalized as a zero-shot text classification task. Instead of using 
    cross-encoder architecture, the bi-encoder (using KeyBERT API) is 
    adopted for performance consideration.
"""

# %%
import os
from pathlib import Path
from typing import List, Union, Any, Dict

import numpy as np
from keybert import KeyBERT


class GoldsteinEventExtractor:
    
    def __init__(self, model: str = 'all-MiniLM-L6-v2', events: List[str] = []):

        self._events = tuple(events)
        self._model = KeyBERT(model)
        self._top_n_events = 4
        self._n_cores = 4

    @property
    def model(self):
        return self._model
    
    @property
    def events(self):
        return self._events
    
    @property
    def top_n_events(self):
        return self._top_n_events
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Regular softmax function, from scores to probabilities"""
        
        ex = np.exp(x - x.max())
        return ex / ex.sum()
    
    # ------------------------------------------------------------------
    def extract(self, texts: Union[List[str], str]) -> List[Dict[str, Any]]:
        """Used to extract most probable Goldstein events from the given sentence/article

        Args:
            txt (str): input sentence such as news article (it can be multiple sentences
                but should be a single string).

        Returns:
            List[Tuple[str, float]]: a list of possible Goldstein events, stored in tuples.
                The float numbers denote the 'likelihood' of that corresponding event.
        """
        
        return self.event_detector.extract_keywords(
            texts, 
            candidates=self.events,
            top_n=self.top_n_events
        )

    def grade(self, text: str) -> float:
        """Grade the event score of the input sentence/article.

        Args:
            txt (str): input sentence such as news article (it can be multiple sentences
                but should be a single string).

        Returns:
            float: Goldstein score
        """
        
        detect = self.extract(text)
        events = np.array([d[0] for d in detect])
        weight = self.softmax(np.array([d[1] for d in detect]))

        # Weighted average
        return self.event_score[events] @ weight
    