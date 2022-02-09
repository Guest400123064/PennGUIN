# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 11-21-2021
# =============================================================================
"""
This module implements helper classes that enables sentence-level tone scoring
    The scoring is based on Hugging Face transformer pre-trained models
"""

# %%
import os
import platform

import pandas as pd
import numpy as np

import torch
import transformers
from keybert import KeyBERT

from typing import Union, Dict, List, Any, Tuple

import spacy
from spacy.tokens import Doc
from spacy.lang.en import English


PG_DOC_EXT_ATTR = 'pg_tone'
GRADER_SOURCE_DIR = os.path.dirname(os.path.realpath(__file__))


def pg_install_extensions() -> None:
    if not Doc.has_extension(PG_DOC_EXT_ATTR):
        Doc.set_extension(PG_DOC_EXT_ATTR, default=[])


class Tone:

    def __init__(self, grader: str, score: float, raw: Any):

        self.grader = grader
        self.score = score
        self.raw = raw

    def __str__(self) -> str:

        ret = f'[ToneScore] :: {self.score} by {self.grader}'
        return ret

    @property
    def data(self):
        return {'grader': self.grader, 'tone': self.score}
    
    
class GoldsteinGrader(spacy.pipeline.Pipe):
    
    link_paper = 'https://www.jstor.org/stable/pdf/174480.pdf'
    col_event = 'EVENT'
    col_score = 'SCORE'
    
    def __init__(self, vocab, name, **kwargs):
        
        self.vocab = vocab
        self.name = name
        pg_install_extensions()
        
        self.grader = 'GoldstineGrader'
        self.lex_score_path = os.path.join(GRADER_SOURCE_DIR, 'goldstein.csv')
        if (self.lex_score_path is None
            or not os.path.exists(self.lex_score_path)):
            raise OSError('[error] @ GoldsteinGrader.__init__() :: ' + 
                f'lexicon score dictionary not found at {self.lex_score_path}')
            
        # Load event tone score
        self.table = pd.read_csv(self.lex_score_path, index_col=GoldsteinGrader.col_event)
        self.table = self.table.loc[:, GoldsteinGrader.col_score]
        self.event = self.table.index.tolist()
        self.model = KeyBERT()
        
    def __call__(self, doc: Doc) -> Doc:
        
        doc._.pg_tone.append(self.grade_single(doc.text))
        return doc

    def _detect_event(self, inputs: str) -> List[Tuple[str, float]]:
        
        ret = self.model.extract_keywords(inputs, candidates=self.event, top_n=5)
        return ret

    def grade_multi(self, inputs: Union[str, List[str]]) -> List[Tone]:
        pass
    
    def grade_single(self, inputs: str) -> Tone:
        
        detect = self._detect_event(inputs)
        events = np.array([d[0] for d in detect])
        weight = np.array([d[1] for d in detect])
        weight = weight / weight.sum()
        
        scores = (self.table[events] * weight).sum()
        return Tone(self.grader, scores, detect)
        


class DistilRobertaFinNewsGrader(spacy.pipeline.Pipe):

    modelcard = 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'

    def __init__(self, vocab, name, **kwargs):

        self.vocab = vocab
        self.name = name
        pg_install_extensions()

        self.grader = 'DistilRobertaFinNewsGrader'
        self.pipeline = transformers.pipeline(
            task='sentiment-analysis',
            model=DistilRobertaFinNewsGrader.modelcard,
            framework='pt',
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1,
            **kwargs
        )

    def __call__(self, doc: Doc) -> Doc:
        
        trf_out = self.pipeline(doc.text)[0]
        doc._.pg_tone.append(self.out2score(trf_out))
        return doc

    def out2score(self, outputs: List[Dict]) -> Tone:
        """Convert transformer outputs to scaler scores.

        Desc:
            The direct output from pipelines are usually stored in dictionaries
                with a `label` field denoting the class name (may vary
                depending on the dataset used for training) and a scaler `score`,
                usually probability. The function aims to normalize the output
                to a unified :obj:`ToneScore`.

        Args:
            out (:obj:`Dict`):
                direct output from the loaded transformer pipeline.

        Returns:
            :obj:`ToneScore`: a unified score interface storing both raw output and
                transformed/normalized tone score.
        """

        try:
            r = 0
            w = {'negative': -1, 'positive': 1, 'neutral': 0}
            for o in outputs:
                s = o['score']
                l = o['label']
                r = r + w[l] * s
        except Exception as e:
            raise e
        return Tone(self.grader, r, outputs)

    def grade(self, inputs: Union[str, List[str]]) -> List[Tone]:
        """Calculate tone score for a sentence or list of sentences."""

        # Multi-thread is not supported for windows; this is a simple
        #   hack to 'suppress' multi-threading on windows
        if platform.system() == 'Windows' and isinstance(inputs, list):
            outputs = [self.pipeline(i)[0] for i in inputs]
        else:
            outputs = self.pipeline(inputs)

        # Normalize to a unified score interface
        scores = [self.out2score(o) for o in outputs]
        return scores


@English.factory('pg-tone-grader')
def make_tone_grader(nlp: English, name: str, grader: str = '', cfg: dict = {}):

    # Currently only one grader, no need for a factory
    return GoldsteinGrader(nlp.vocab, name, **cfg)

# %%
