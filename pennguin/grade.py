# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 02-10-2022
# =============================================================================
"""
This module implements event graders that takes an event grading schema (such as 
    a scoring table that maps each event to a score) and a event extractor
    that extract events from given text along with confidence scores. 
    Then, the grader will calculate an average 'tone' grade weighted 
    by confidence scores.
"""

# %%
from typing import Dict, List, Union, Any
from .event_extract import BaseEventExtractor, KeyBERTEventExtractor


class GoldsteinGrader:

    def __init__(
        self, 
        goldstein_grade: Dict[str, float],
        event_extractor: BaseEventExtractor
    ):
        self.grader = lambda e: goldstein_grade[e]
        self.events = list(goldstein_grade.keys())
        self.extractor = event_extractor

    # -------------------------------------------------------------
    def grade(self, texts: Union[List[str], str]) -> List[float]:
        
        if isinstance(texts, str):
            return [self._grade_single(texts)]
        elif isinstance(texts, list):
            return self._grade_multi(texts)
        else:
            raise ValueError('@ GoldsteinGrader.extract() :: ' + 
                f'Invalid <texts> type {type(texts)}; only <str, List[str]> allowed')
    
    
    def _grade_extract(self, extract: Dict[str, Any]) -> float:
        
        ret = zip(extract['events'], extract['scores'])
        src = sum((self.grader(e) * s) for (e, s) in ret)
        return src


    def _grade_single(self, text: str) -> float:
        
        extract = self.extractor.extract(text, self.events)[0]
        return self._grade_extract(extract)
    
    
    def _grade_multi(self, texts: List[str]) -> List[float]:
        
        extract = self.extractor.extract(texts, self.events)
        return [self._grade_extract(ex) for ex in extract]

# %%
# Sample usage
if __name__ == '__main__':
    
    import json
    
    # Load grade table
    with open('goldstein.json') as f:
        goldstein = json.load(f)
        
    # Init a grader
    extractor = KeyBERTEventExtractor()
    grader = GoldsteinGrader(goldstein, extractor)
    
    # Grading
    text = [
        'There is a war between A and B and several officials are recalled.',
        'What a nice day!',
        'A and B meet and discussed about how to resolve the current crisis.'
    ]
    print(grader.grade(text))

# %%
