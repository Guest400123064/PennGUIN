# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 02-09-2022
# =============================================================================
"""
This module implements helper classes to extract events. The problem 
    is formalized as a zero-shot text classification task. 
    
There are two categories of models: 
    - Cross-encoder: potentially high acc but low speed (when too many possible events).
    - Bi-encoder: potentially low acc but high speed.
"""

# %%
from typing import List, Union, Any, Dict
from abc import ABC, abstractmethod

import numpy as np

from keybert import KeyBERT
from transformers import pipeline


class BaseEventExtractor(ABC):
    
    @abstractmethod
    def extract(self, texts: Union[List[str], str], events: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class KeyBERTEventExtractor(BaseEventExtractor):
    
    def __init__(
        self, 
        model: str = 'all-MiniLM-L6-v2', 
        top_n_events: int = 4, 
        temperature: float = 0.1
    ):
        self._model = KeyBERT(model)
        self._top_n_events = top_n_events
        self._temperature = temperature

    @property
    def model(self):
        return self._model
    
    @property
    def top_n_events(self):
        return self._top_n_events

    def softmax(self, x: np.ndarray) -> List[float]:
        """Regular softmax function, from scores to probabilities. 
            The temperature parameter <self._temperature> is used to 
            sharpen the distribution. The smaller the value, the 
            sharper the distribution (closer to 'hard max/normal max')"""
        
        ex = np.exp((x - x.max()) / self._temperature)
        return (ex / ex.sum()).tolist()
    
    # ------------------------------------------------------------------
    def extract(self, texts: Union[List[str], str], events: List[str] = ['[NULL]']) -> List[Dict[str, Any]]:
        """Used to extract most probable events from the given sentence/article

        Args:
            texts (Union[List[str], str]): input sentences such as news article.
            events (List[str]): list of possible events

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
            return [extractor(texts, events)]
        elif isinstance(texts, list):
            return [extractor(t, events) for t in texts]
        else:
            raise ValueError('@ KeyBERTEventExtractor.extract() :: ' + 
                f'Invalid <texts> type {type(texts)}; only <str, List[str]> allowed')
    
    
    def _extract_single(self, text: str, events: List[str] = ['[NULL]']) -> Dict[str, Any]:
        """Driver used to extract events from a single article/sentence"""

        extract, scores = zip(
            *self.model.extract_keywords(
                text, candidates=events, top_n=self.top_n_events
            )
        )
        return {
            'text': text,
            'events': list(extract),
            'scores': self.softmax(np.array(scores)),
            'cosine': list(scores)
        }
        
        
class BARTEventExtractor(BaseEventExtractor):
    
    def __init__(self, model: str = 'facebook/bart-large-mnli', top_n_events: int = 4):

        self._model = pipeline('zero-shot-classification', model)
        self._top_n_events = top_n_events


class SentBERTEventExtractor(BaseEventExtractor):
    
    def __init__(self, model: str = 'cross-encoder/nli-MiniLM2-L6-H768', top_n_events: int = 4):

        self._model = pipeline('zero-shot-classification', model)
        self._top_n_events = top_n_events

# %%