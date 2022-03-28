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
    extractor = KeyBERTEventExtractor(temperature=0.1)
    grader = GoldsteinGrader(goldstein, extractor)
    
    # Grading
    text = [
        'Disagreements about labor practices between A and B have stalled trade negotiations.',
        'Relations between A and B are beset by a minefield of disputes across a wide range of issue areas.',
        'The convention brought together A and B to discuss the mining project\'s environmental impact.',
        'A and B failed to resolve their disputes across a wide range of issue areas.',
        'There was disagreement between A and B on which labor practices to implement.',
        'The disagreement between A and B on labor practices delayed progress in the talks.',
        "France is accused of missing or ignoring the warning signs for the 1994 Rwanda massacre, and sending troops only to counter the Tutsi rebels of Paul Kagame, who is now president The dispute centres on France's role prior to the genocide as a close ally of the Hutu nationalist regime of Juvenal Habyarimana."
    ]
    print(grader.grade(text))

# %%
