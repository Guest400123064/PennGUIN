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


from typing import Dict, List, Callable
from .event_extract import BaseEventExtractor


class GoldsteinGrader:
    
    def __init__(
        self, 
        grade_fn: Dict[str, float], 
        event_list: List[str],
        event_extractor: BaseEventExtractor
    ):
        self.grader = grade_fn
        self.event_list = event_list
        self.extractor = event_extractor