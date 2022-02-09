# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Email: wangy49@seas.upenn.edu
# Date: 02-09-2022
# =============================================================================
"""
This module implements helper class that can help scraping articles from a 
    given list of urls.
"""


from ssl import SSLError
from typing import List, Union, Tuple
from multiprocessing import Pool as ThreadPool

import requests
from bs4 import BeautifulSoup

import trafilatura
from trafilatura.settings import use_config


# Error labels
EMPTY_RETRIEVAL = '[EMPTY_RETRIEVAL]'
ERROR_STATUS_CODE = {
    429, 499, 500, 502, 503, 504, 509, 
        520, 521, 522, 523, 524, 525, 526, 527, 530, 598
}


class NewsScraper:
    
    def __init__(
        self, 
        fetcher: str  = 'request',
        verbose: bool = True
    ) -> None:
        
        self.fetcher = fetcher
        self.verbose = verbose
        self.threads = 4
    
    
    def html2doc(self, html: str) -> str:
        
        ret = trafilatura.extract(
            html, 
            favor_precision=True,
            include_comments=False
        )
        return ret
    
    
    def fetch(self, urls: Union[List[str], str]) -> List[Tuple[str, str]]:
        
        # Backend selection
        if self.fetcher == 'request':
            fetcher = self._fetch_req
        elif self.fetcher == 'trafilatura':
            raise NotImplementedError('NOT YET IMPLEMENTED')

        # Direct retrieval for single url
        if isinstance(urls, str):
            return [fetcher(urls)]
        elif isinstance(urls, list):
            with ThreadPool(self.threads) as p:
                return p.map(fetcher, urls)
        else:
            raise ValueError(f'Unknown <urls> type {type(urls)}; ' + 
                    'only <str, List[str]> allowed')

    
    def _fetch_req(self, url: str) -> Tuple[str, str]:
        
        head = {'User-Agent': 'Mozilla/5.0'}
        try:
            resp = requests.get(url, headers=head)
            
        # Try no_ssl fetch
        except SSLError:
            try:
                resp = requests.get(url, headers=head, verify=False)
            except Exception as e:
                return (None, EMPTY_RETRIEVAL)
    
        # Any error status
        if resp.status_code in ERROR_STATUS_CODE:
            return (None, EMPTY_RETRIEVAL)
        
        # Parse and return
        text = str(BeautifulSoup(resp.content, 'html5lib'))
        if text is None:
            return (None, EMPTY_RETRIEVAL)
        return (text, self.html2doc(text))
    
    
    def _fetch_tra(self, url: str) -> Tuple[str, str]:
        
        return (None, EMPTY_RETRIEVAL)
