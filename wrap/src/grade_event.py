# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 05-27-2022
# =============================================================================
"""
This module implements helper classes to detect events. The problem 
    is formalized as a zero-shot-text-classification task. Specifically, given
    any input text that (potentially) describes some events, we try to classify 
    which particular event the input text is describing.
    
There are two categories of models: 
    - Cross-encoder: potentially high acc but low speed (when too many possible events).
    - Bi-encoder: potentially low acc but high speed. 
    
Note that the current implementation of the Bi-encoder extractor relies
    on KeyBERT backend which DO NOT have caching or batch-processing mechanisms. Thus, 
    the speed for running the two backends are similar. In the future, we may implement 
    batch processing pipelines directly using SentenceBERT.
    
Aside from event extractor, there is also a GoldsteinGrader class. It takes Goldstein 
    Event Scale <http://web.pdx.edu/~kinsella/jgscale.html> and a 
    event extractor. Then, the grader will calculate an average 'tone' grade weighted 
    by confidence scores (likelihood of event being mentioned).
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
from keybert import KeyBERT
from transformers import pipeline
from datasets import Dataset


class BaseEventExtractor(ABC):
    
    @abstractmethod
    def extract(self, texts: Union[List[str], str], events: List[str]) -> List[Dict[str, Any]]:
        """Used to extract most probable events from the given sentence/article

        Args:
            texts (Union[List[str], str]): input sentences such as news article.
            events (List[str]): list of possible events

        Raises:
            ValueError: invalid article input type.

        Returns:
            List[Dict[str, Any]]: extracted events stored in dictionaries of format:
                
                {
                    'text': <input text>,
                    'events': <list of top possible events>,
                    'std_scores': <(Normalized) The float numbers denote the 'likelihood' of that corresponding event>,
                    'raw_scores': <Raw predictions directly from the model>
                }
        """
        raise NotImplementedError
    
    def softmax(self, x: np.ndarray, t: float) -> List[float]:
        """Regular softmax function, from scores to probabilities. 
            The temperature parameter <t> is used to 
            sharpen the distribution. The smaller the value, the 
            sharper the distribution (closer to 'hard max/normal max')"""
        
        ex = np.exp((x - x.max()) / t)
        return (ex / ex.sum()).tolist()
    
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


class KeyBERTEventExtractor(BaseEventExtractor):
    
    def __init__(
        self, 
        model: str = 'all-mpnet-base-v2', 
        top_n_events: int = 4, 
        temperature: float = 0.03
    ):
        self._model_card = model
        self._model = KeyBERT(model)
        self._top_n_events = top_n_events
        self._temperature = temperature
        
    def __repr__(self):
        return f'KeyBERTEventExtractor("{self.model_card}", {self.top_n_events}, {self.temperature})'
        
    @property
    def model_card(self):
        return self._model_card

    @property
    def model(self):
        return self._model
    
    @property
    def top_n_events(self):
        return self._top_n_events
    
    @property
    def temperature(self):
        return self._temperature
    
    # ------------------------------------------------------------------
    def extract(self, texts: Union[List[str], str], events: List[str] = ['[NULL]']) -> List[Dict[str, Any]]:
        """Implementation using KeyBERT backend"""
        
        extractor = self._extract_single
        
        # Direct extraction for single article
        if isinstance(texts, str):
            texts = self.preprocess(texts)
            return [extractor(texts, events)]
        elif isinstance(texts, list):
            texts = [self.preprocess(t) for t in texts]
            return [extractor(t, events) for t in texts]
        else:
            raise ValueError('@ KeyBERTEventExtractor.extract() :: ' + 
                f'Invalid <texts> type {type(texts)}; only <str, List[str]> allowed')
    
    def _extract_single(self, text: str, events: List[str] = ['[NULL]']) -> Dict[str, Any]:
        """Driver used to extract events from a single article/sentence"""

        extract, scores = zip(*self.model.extract_keywords(text, candidates=events, top_n=self.top_n_events))
        return {
            'text': text,
            'events': list(extract),
            'std_scores': self.softmax(np.array(scores), self.temperature),
            'raw_scores': list(scores)
        }


class HuggingfaceZeroShotEventExtractor(BaseEventExtractor):
    
    def __init__(
        self, 
        model: str = 'valhalla/distilbart-mnli-12-1', 
        top_n_events: int = 4,
        temperature: float = 0.1,
        batch_size: int = 32
    ):
        self._model_card = model
        self._model = pipeline(
            'zero-shot-classification', model, 
            device=(0 if torch.cuda.is_available() else -1)  # Using the first GPU only
        )
        self._top_n_events = top_n_events
        self._temperature = temperature
        self._batch_size = batch_size
        
    def __repr__(self):
        return f'HuggingfaceZeroShotEventExtractor("{self.model_card}", {self.top_n_events}, {self.temperature}, {self.batch_size})'
        
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
    def top_n_events(self):
        return self._top_n_events
    
    @property
    def temperature(self):
        return self._temperature
    
    @property
    def batch_size(self):
        return self._batch_size
    
    # -----------------------------------------------------------------------------------
    def extract(self, texts: Union[List[str], str], events: List[str] = ['[NULL]']) -> List[Dict[str, Any]]:
        """Implementation using Huggingface ZeroShotClassification pipeline as the backend"""
        
        # Direct extraction for single article
        if isinstance(texts, str):
            texts = self.preprocess(texts)
            return [self._extract_single(texts, events)]
        elif isinstance(texts, list):
            texts = [self.preprocess(t) for t in texts]
            return self._extract_batch(texts, events)
        else:
            raise ValueError('@ HuggingfaceZeroShotEventExtractor.extract() :: ' + 
                f'Invalid <texts> type {type(texts)}; only <str, List[str]> allowed')
    
    def _extract_single(self, text: str, events: List[str] = ['[NULL]']) -> Dict[str, Any]:
        """Driver used to extract events from a single article/sentence"""

        output = self.model(text, events)
        scores = np.array(output['scores'][:self.top_n_events])
        scores = scores / scores.sum()
        return {
            'text': text,
            'events': output['labels'][:self.top_n_events],
            'std_scores': self.softmax(scores, self.temperature),
            'raw_scores': scores.tolist()
        }
        
    def _extract_batch(self, texts: List[str], events: List[str] = ['[NULL]']) -> List[Dict[str, Any]]:
        """Batch processing version of extract single"""
        
        # Helper function for batched inference
        def _predict(batch, pipe, labels):
            return {'outputs': pipe(batch['texts'], candidate_labels=labels)}
        
        # Create a huggingface dataset for batched processing
        outputs = (Dataset
            .from_dict({'texts': texts})
            .map(_predict, batched=True, batch_size=self.batch_size, 
                    fn_kwargs={'labels': events, 'pipe': self.model})
            .to_dict()
            .get('outputs'))
        
        # Cut off to top-n-events & normalize scores
        for i, o in enumerate(outputs):
            scores = np.array(o['scores'][:self.top_n_events])
            scores = scores / scores.sum()
            outputs[i] = {
                'text': o['sequence'],
                'events': o['labels'][:self.top_n_events],
                'std_scores': self.softmax(scores, self.temperature),
                'raw_scores': scores.tolist()
            }
        return outputs
    
    
class GoldsteinGrader:
    
    def __init__(self, goldstein_grade: Dict[str, float], event_extractor: BaseEventExtractor):
        self._grade_dict = goldstein_grade
        self._grader = lambda e: goldstein_grade[e]
        self._events = list(goldstein_grade.keys())
        self._extractor = event_extractor

    @property
    def extractor(self):
        return self._extractor
    
    @property
    def events(self):
        return self._events
    
    @property
    def grade_dict(self):
        return self._grade_dict
    
    @property
    def grader(self):
        return self._grader
        
    # -------------------------------------------------------------
    def grade(self, texts: Union[List[str], str]) -> List[Dict[str, Any]]:
        """Grade (co-mention) texts based on possible events involved.

        Args:
            texts (Union[List[str], str]): input sentences such as news article.

        Raises:
            ValueError: invalid article input type.

        Returns:
            List[Dict[str, Any]]: extracted events and Goldstein scores stored in dictionaries of format:
                
                {
                    'text': <input text>,
                    'events': <list of top possible events>,
                    'std_scores': <(Normalized) The float numbers denote the 'likelihood' of that corresponding event>,
                    'raw_scores': <Raw predictions directly from the model>,
                    'goldstein': <Goldstein score>
                }
        """
        
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, list):
            texts = texts
        else:
            raise ValueError('@ GoldsteinGrader.grade() :: ' + 
                f'Invalid <texts> type {type(texts)}; only <str, List[str]> allowed')
    
        # Detect events and grade based on scale table
        extract = self.extractor.extract(texts, self.events)
        return [self._calc_score(ex) for ex in extract]
    
    def _calc_score(self, extract: Dict[str, Any]) -> float:
        """Calculate weighted average of the event `sentiment scores` according to the 
            Goldstein Scale and confidence of the event detected (i.e., how likely is the
            given text mentioning a particular event)"""
        
        ret = zip(extract['events'], extract['std_scores'])
        extract.update({'goldstein': sum((self.grader(e) * c) for (e, c) in ret)})
        return extract


# Sample usage
if __name__ == '__main__':
    
    sample_texts = [
        'Xi who was flanked by his wife Peng Liyuan was received yesterday by President Paul Kagame and his wife his wife Jeannette Kagame.',
        'Nation World. Rwanda somberly marks the start of genocide 25 years ago. President Paul Kagame and first lady Jeannette Kagame laid wreaths and lit a flame at the mass burial ground of 250,000 victims at the Kigali Genocide Memorial Center in the capital, Kigali.',
        'There is broad agreement that this success is attributable in large measure to the strong leadership and dedication of President Paul Kagame and the First Lady of Rwanda, Mrs Jeannette Kagame through her Imbuto Foundation, all underpinned by robust and innovative Government programmes, broad involvement of all the stakeholders and communities as well as adequate external support.',
        'Disagreements about labor practices between A and B have stalled trade negotiations.',
        'Relations between A and B are beset by a minefield of disputes across a wide range of issue areas.',
        'The convention brought together A and B to discuss the mining project\'s environmental impact.',
        'A and B failed to resolve their disputes across a wide range of issue areas.',
        'There was disagreement between A and B on which labor practices to implement.',
        'The disagreement between A and B on labor practices delayed progress in the talks.',
        "France is accused of missing or ignoring the warning signs for the 1994 Rwanda massacre, and sending troops only to counter the Tutsi rebels of Paul Kagame, who is now president The dispute centres on France's role prior to the genocide as a close ally of the Hutu nationalist regime of Juvenal Habyarimana."
    ]
    
    # Load grade table
    with open('../models/goldstein.json') as f:
        goldstein = json.load(f)
    
    # Grade sentences using Goldstein scales
    extractor = HuggingfaceZeroShotEventExtractor(top_n_events=3)
    grader = GoldsteinGrader(goldstein, extractor)
    pprint(grader.grade(sample_texts))

# %%
