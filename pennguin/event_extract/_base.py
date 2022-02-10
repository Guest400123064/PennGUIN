# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 02-10-2022
# =============================================================================
"""
The base class of event extractors.
"""


from typing import List, Union, Any, Dict
from abc import ABC, abstractmethod


class BaseEventExtractor(ABC):
    
    @abstractmethod
    def extract(self, texts: Union[List[str], str], events: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError
