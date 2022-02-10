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

# %%
import logging
from typing import List, Union, Tuple
from multiprocessing import Pool as ThreadPool

import requests
from bs4 import BeautifulSoup
from ssl import SSLError

import trafilatura


# Error labels
EMPTY_TEXT = '[EMPTY_TEXT]'
EMPTY_HTML = '[EMPTY_HTML]'
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
        self.n_cores = 4
    
    
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
            with ThreadPool(self.n_cores) as p:
                return p.map(fetcher, urls)
        else:
            raise ValueError('@ NewsScraper.fetch() :: ' + 
                f'Invalid <urls> type {type(urls)}; only <str, List[str]> allowed')

    
    def _fetch_req(self, url: str) -> Tuple[str, str]:
        
        head = {'User-Agent': 'Mozilla/5.0'}
        try:
            resp = requests.get(url, headers=head)
            
        # Try no_ssl fetch
        except SSLError:
            try:
                resp = requests.get(url, headers=head, verify=False)
            except Exception as e:
                logging.warning('@ NewsScraper.fetch() :: ' + 
                    f'failed to fetch from url <{url}>: {e}')
                return (EMPTY_HTML, EMPTY_TEXT)
        
        # Other exception, such as invalid url
        except Exception as e:
            logging.warning('@ NewsScraper.fetch() :: ' + 
                f'failed to fetch from url <{url}>: {e}')
            return (EMPTY_HTML, EMPTY_TEXT)

        # Any error status
        if resp.status_code in ERROR_STATUS_CODE:
            return (EMPTY_HTML, EMPTY_TEXT)
        
        # Parse and return
        html = str(BeautifulSoup(resp.content, 'html5lib'))
        if html is None:
            return (None, EMPTY_TEXT)
        return (html, self.html2doc(html))
    
    
    def _fetch_tra(self, url: str) -> Tuple[str, str]:
        
        return (EMPTY_HTML, EMPTY_TEXT)

# %%
# Sample usage
if __name__ == '__main__':
    
    # Init scraper and fetch
    scraper = NewsScraper()
    html, text = scraper.fetch('https://github.blog/2019-03-29-leader-spotlight-erin-spiceland/')[0]
    
    # Print results
    print(text)
    print(html)
