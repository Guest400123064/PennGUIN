# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 11-21-2021
# =============================================================================


# %%
import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc
from spacy.vocab import Vocab

from typing import Any, List, Tuple, Union, Dict
from .pattern import pattern, BasePattern


PG_DOC_EXT_ATTR = 'pg_mention'


def pg_install_extensions() -> None:
    if not Doc.has_extension(PG_DOC_EXT_ATTR):
        Doc.set_extension(PG_DOC_EXT_ATTR, default=None)


class MentionDetector(spacy.pipeline.Pipe):

    def __init__(
        self,
        vocab: Vocab,
        name: str = 'mention_detector',
        entities: List[Tuple[str, Dict[str, Any]]] = []
    ) -> None:

        self.name = name
        self.vocab = vocab
        pg_install_extensions()

        self.entities = entities
        self.patterns = [pattern(e[0], e[1]) for e in entities]
        self.matchers = [self._make_matcher(p) for p in self.patterns]

    def _make_matcher(self, pattern: BasePattern) -> Matcher:

        matcher = Matcher(self.vocab)
        matcher.add(str(pattern), pattern.pattern)
        return matcher

    def __call__(self, doc: Doc) -> Doc:

        mention_list = []
        for matcher, pattern in zip(self.matchers, self.patterns):
            result = matcher(doc)
            if len(result) > 0:
                mention_list.append(pattern)
            doc._.pg_mention = mention_list
        return doc


@Language.factory('pg-mention-detector')
def make_mention_detector(nlp: Language, name: str, entities: List[Tuple[str, Dict[str, Any]]]):
    return MentionDetector(nlp.vocab, name, entities)
