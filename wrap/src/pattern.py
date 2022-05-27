# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 11-23-2021
# =============================================================================
"""
This module implements the patterns that help detect whether certain entities 
    are mentioned within a given piece of text (usually a sentence)
"""

# %%
from typing import List
from abc import ABC, abstractproperty
from nameparser import HumanName


class BasePattern(ABC):
    """Base class for all patterns. It ensures that all derived patterns must have 
    
    Properties:
        pattern (List[List[dict]]): `spacy.Matcher` readable pattern
        uid (int): an id to identify the entity to correspond with other meta information or for disambiguation
        label (str): a human redable short label
        data (dict): a collection of data (can be any according to the entity)
    """

    @abstractproperty
    def uid(self):
        raise NotImplementedError
    
    @abstractproperty
    def label(self):
        raise NotImplementedError

    @abstractproperty
    def pattern(self) -> List[List[dict]]:
        """The pattern is in a list-of-lists-of-dictionaries structure. 
        
        Each one of the internal list is a list of dictionaries. The entire list 
            represent A COLLECTION of patterns of one entity. In other words,
            each internal list is a potential pattern of the entity interested.
            Any one hit result in a hit: [A or B or ... or Z].
            
        Each dictionary corresponds to a TOKEN, not the entity. If an entity is
            expected to have two tokens, e.g., human name, then one of the internal 
            list will have two dictionaries that try to mach first name and last name
            respectively. However, if first-name-only is allowed, then there can be 
            an internal list that only have one dictionary.
            
        Example:
            To match a person <Morty Smith>:
                [
                    [{'LOWER': 'morty'}, {'LOWER': 'smith'}],  # Pattern 1
                    [{'LOWER': 'morty'}]                       # Pattern 2
                ]
        """
        raise NotImplementedError
    
    @abstractproperty
    def data(self) -> dict:
        raise NotImplementedError

    
class PersonPattern(BasePattern):
    """Used to generate patterns for human names"""
    
    ENT_TYPE = 'PERSON'
    
    def __init__(self, uid: int, name: str, **kwargs) -> None:
        """Initialize a <human name> to <pattern> convertor. An `id` and a (human) `name` is 
        mandatory for initialization. 

        Args:
            uid (int): unique id to correspond database
            name (str): human name
        """

        self._id = uid
        self._lb = name
        self._hn = HumanName(name)
        self._kw = kwargs
        self._cfg = kwargs.get('cfg')

    def __repr__(self) -> str:
        return f'PersonPattern(uid={self.uid}, name="{self.label}")'

    def __str__(self) -> str:
        return str(self.data)

    @property
    def first(self) -> dict:
        """Generate token pattern to match first name"""

        ret = {
            'LOWER': self._hn.first.lower(),
            'ENT_TYPE': PersonPattern.ENT_TYPE
        }
        return ret

    @property
    def last(self) -> dict:
        """Generate token pattern to match last name"""

        ret = {
            'LOWER': self._hn.last.lower(),
            'ENT_TYPE': PersonPattern.ENT_TYPE
        }
        return ret
    
    @property
    def uid(self):
        return self._id
    
    @property
    def label(self):
        return self._lb

    @property
    def pattern(self) -> List[List[dict]]:
        """Ensemble patterns according to matching rules. Currently 
        the `self._cfg` is only a place holder.

        Returns:
            List[List[dict]]: generate `spacy.matcher.Matcher` compatible patterns
        """

        ret = [[self.first, self.last], [self.first]]
        return ret
    
    @property
    def data(self) -> dict:
        return {
            'type': PersonPattern.ENT_TYPE,
            'uid': self.uid,
            'label': self.label
        }


class CompanyPattern(BasePattern):
    """Used to generate patterns for companies, currently a place holder"""
    pass


def pattern(ent_type: str, ent_data: dict) -> BasePattern:
    
    types = {
        'PERSON': PersonPattern,
        'COMPANY': CompanyPattern
    }
    ent = types.get(ent_type)
    if ent is None:
        raise KeyError('[error] @ pattern.patter() :: ' + 
            f'unknown entity type <{ent_type}>') 
    return ent(**ent_data)

