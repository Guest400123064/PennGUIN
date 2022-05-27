# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 02-09-2022
# =============================================================================
"""
This module implements an AllenNLP backend for co-reference resolution
"""

# %%
from typing import List, Union

from torch.cuda import is_available as is_cuda_available
from allennlp_models.coref.predictors.coref import CorefPredictor


MODEL_CARD = r'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz'
CUDA_AVAIL = 0 if is_cuda_available() else -1


class AllenCorefPredictor:
    
    def __init__(self, model: str = MODEL_CARD):
        """A simple wrapper of pre-trained `allennlp` coref resolution 
            model. The main method `resolve` will take list of 
            raw articles and substitute all possible references such 
            as 'His', 'I', 'Her', etc. Note that there can be potential 
            mis-substitutions.
        """

        self._model = CorefPredictor.from_path(model, cuda_device=CUDA_AVAIL)
        self._spacy = self.model._spacy

    @property
    def model(self):
        return self._model
    
    @property
    def spacy(self):
        return self._spacy
        
    # -------------------------------------------------------------------------
    def resolve(self, texts: Union[List[str], str]) -> List[str]:
        """API for co-reference resolution using `allennlp` backend. 
            Given a single or a list of articles, perform coref resolution 
            with pre-trained model by AllenNLP.

        Args:
            texts (Union[List[str], str]): [description]

        Raises:
            ValueError: when `texts` is not a valid argument

        Returns:
            List[str]: resolved list of articles for coref resolution
        """
        
        if isinstance(texts, str):
            return [self.model.coref_resolved(texts)]
        elif isinstance(texts, list):
            return self._resolve_multi(texts)
        else:
            raise ValueError('@ AllenCorefPredictor.fetch() :: ' + 
                f'Unknown <texts> type {type(texts)}; only <str, List[str]> allowed')
        
    def _resolve_multi(self, texts: List[str]) -> List[str]:
        """A batch processing wrapper for `CorefPredictor.resolve`

        Args:
            texts (List[str]): list of articles for coref resolution

        Returns:
            List[str]: resolved list of articles for coref resolution
        """
        
        document = self.spacy.pipe(texts)
        clusters = [pred.get('clusters') for pred in
            self.model.predict_batch_json({'document': s} for s in texts)]

        return [self.model.replace_corefs(d, c) for d, c in zip(document, clusters)]

# %%
# Sample usage
if __name__ == '__main__':
    
    # Init resolver; it can take a while to load
    model = AllenCorefPredictor()
    
    # Substitution
    text = "Eva and Martha didn't want their friend Jenny to feel lonely so they invited her to the party in Las Vegas."
    resolved = model.resolve(text)[0]
    
    print(text)
    print(resolved)
