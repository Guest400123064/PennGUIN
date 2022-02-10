# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 02-09-2021
# =============================================================================
"""
This module implements helper classes to extract events. The problem 
    is formalized as a zero-shot text classification task. Instead of using 
    cross-encoder architecture, the bi-encoder (e.g. using KeyBERT API) is 
    adopted for performance consideration.
"""

# %%
from typing import List, Union, Any, Dict
from multiprocessing import Pool as ThreadPool

import numpy as np
from keybert import KeyBERT


class KeyBERTEventExtractor:
    
    def __init__(self, model: str = 'all-MiniLM-L6-v2', events: List[str] = []):

        self._events = tuple(events)
        self._model = KeyBERT(model)
        self._top_n_events = 4
        self.n_cores = 4

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
    def softmax(x: np.ndarray) -> List[float]:
        """Regular softmax function, from scores to probabilities"""
        
        ex = np.exp(x - x.max())
        return (ex / ex.sum()).tolist()
    
    # ------------------------------------------------------------------
    def extract(self, texts: Union[List[str], str]) -> List[Dict[str, Any]]:
        """Used to extract most probable events from the given sentence/article

        Args:
            texts (str): input sentence such as news article (it can be multiple sentences
                but should be a single string).

        Raises:
            ValueError: Invalid article input type.

        Returns:
            List[Dict[str, Any]]: extracted events stored in dictionaries of format:
                
                {
                    'events': <list of top possible events>,
                    'scores': <The float numbers denote the 'likelihood' of that corresponding event>
                }
        """
        
        extractor = self._extract_single
        
        # Direct extraction for single article
        if isinstance(texts, str):
            return [extractor(texts)]
        elif isinstance(texts, list):
            with ThreadPool(self.n_cores) as p:
                return p.map(extractor, texts)
        else:
            raise ValueError('@ KeyBERTEventExtractor.extract() :: ' + 
                f'Invalid <texts> type {type(texts)}; only <str, List[str]> allowed')
    
    
    def _extract_single(self, text: str) -> Dict[str, Any]:
        """Driver used to extract events from a single article/sentence"""

        events, scores = zip(
            *self.event_detector.extract_keywords(
                text, candidates=self.events, top_n=self.top_n_events
            )
        )
        return {
            'events': events,
            'scores': self.softmax(np.array(scores))
        }
