# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 11-21-2021
# =============================================================================
"""
This module implements helper class that can help scraping articles from a 
    given list of urls (stored in csv). The scraping classes maintains a 
    compatible interface with sklearn transformer for easy integration into 
    future piplines.
"""

# %%
import os
from pathlib import Path

# For text normalization
import re
import unicodedata

# Scraping backends
import requests
from bs4 import BeautifulSoup

import trafilatura
from trafilatura.settings import use_config

# For uniformed interface
from typing import List, Tuple, Dict, Union, Iterable

# Error labels
EMPTY_RETRIEVAL = 'EMPTY_RETRIEVAL'
ERROR_STATUS_CODE = {
    429, 499, 500, 502, 503, 504, 509, 
        520, 521, 522, 523, 524, 525, 526, 527, 530, 598
}


class Scraper:

    def __init__(self, config: str=None, verbose=False) -> None:

        self.verbose = verbose

        # TODO: the current config is only a placeholder
        #   trafilatura.fetch_url() sometimes failed (cannot terminate). Thus
        #   use manual url fetch for now.
        if config is None:
            self.config = use_config(str(Path(__file__).parent / 'trafilatura.cfg'))
        else:
            if not os.path.exists(config):
                raise FileNotFoundError('[error] @ TrafilaturaScraper.__init__() :: ' + 
                    f'config file at <{config}> not found')
            self.config = use_config(config)

    def __call__(self, inputs: Union[Iterable[str], str]) -> List[str]:
        """Make the scraper callable, iterate through given list of urls and extract contents

        Args:
            inputs (Union[Iterable[str], str]): url(s)

        Raises:
            TypeError: non-accepted input url

        Returns:
            List[str]: list of retrieved articles as string
        """
        
        if isinstance(inputs, str):
            return [self.get_single(inputs)]
        elif isinstance(inputs, Iterable):
            return self.get_multi(inputs)
        else:
            raise TypeError('[error] @ TrafilaturaScraper.__call__() :: ' +
                f'str or List[str] inputs expected; {type(inputs)} found')

    def get_multi(self, inputs: Iterable[str]) -> List[str]:
        """A simple wrapper for multiple urls

        Args:
            inputs (Iterable[str]): input urls

        Returns:
            List[str]: list of retrieved articles as string
        """
        
        return [self.get_single(url) for url in inputs]

    def get_single(self, inputs: str) -> str:
        """Driver function used to retrieve text contents from a single given url

        Args:
            inputs (str): [description]

        Raises:
            TypeError: input url should only be string

        Returns:
            str: retrieved articles as string
        """

        if not isinstance(inputs, str):
            raise TypeError('[error] @ TrafilaturaScraper.get_single() :: ' + 
                f'input url should be str; {type(inputs)} found')

        # TODO: use log module in the future
        if self.verbose:
            print('[info] @ TrafilaturaScraper.get_single() :: ' + 
                f'extracting content from <{inputs}>')

        # Manual URL fetch because cannot control retry behavior 
        #   through trafilatura API
        try:
            resp = requests.get(inputs, headers={'User-Agent': 'Mozilla/5.0'})
        except Exception as e:
            print('[warn] @ TrafilaturaScraper.get_single() :: ' +
                f'failed to retrieve data from <{inputs}> with error:')
            print(f'[warn] >>> {e}')
            return EMPTY_RETRIEVAL

        if resp.status_code in ERROR_STATUS_CODE:
            return EMPTY_RETRIEVAL

        # Parse html and extract
        soup = BeautifulSoup(resp.content, 'html5lib')
        text = trafilatura.extract(str(soup))
        if text is None:
            return EMPTY_RETRIEVAL

        return self.normalize(text)
    
    def normalize(self, txt: str) -> str:
        """Help clean the scraped articles by removing extra spaces etc.

        Args:
            txt (str): raw retrieved article text

        Returns:
            str: normalized text
        """

        txt = unicodedata.normalize('NFKD', txt)
        txt = re.sub(r'\s+', ' ', txt.strip())
        return txt

# %%
