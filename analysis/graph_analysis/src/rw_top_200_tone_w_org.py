# %%
from typing import List, FrozenSet, Tuple, Any, Dict
from collections import defaultdict

import pandas as pd
import numpy as np

import igraph as ig
import leidenalg as la
from pyvis.network import Network as VisNetwork

SRC_EDGE_FN = '/home/lgfz1/Projects/pennguin/analysis/graph_analysis/data/rw_top_200_tone_w_org.csv'


# Load raw co-mention edge list and merge accross articles
#   IMPORTANT: Though there are fields <co_mentions_sum> 
#       and <co_mentions_count> in the source edge list file, 
#       they are PER ARTICLE statistics. So we still need 
#       to sum them up individually and manually calculate
#       average
df_edge = df_edge = (
    pd.read_csv(SRC_EDGE_FN)
        .dropna(subset='id2')
        .groupby(['id1', 'id2', 'flag_person', 'flag_company'], as_index=False, sort=False)
            [['co_mentions_sum', 'co_mentions_count']]
        .sum()
)

# Calculate average tone and confidence of estimation (log10 count and normalize to [0, 1])
df_edge.loc[:, 'score_average']          = df_edge.co_mentions_sum / df_edge.co_mentions_count
df_edge.loc[:, 'score_confidence']       = np.log10(df_edge.co_mentions_count)
df_edge.loc[:, 'score_confidence']       = df_edge.score_confidence / df_edge.score_confidence.max()
df_edge.loc[:, 'score_average_weighted'] = df_edge.score_average * df_edge.score_confidence

# Filter out people-only edges
mask_ppl = np.logical_and(
    df_edge.loc[:, 'flag_person'] == 1, 
    df_edge.loc[:, 'flag_company'] == 0
)
df_edge_ppl = df_edge.loc[mask_ppl]
df_edge_org = df_edge.loc[~mask_ppl]

# %%
# Build Igraph Network
class PeopleNet:
    
    def __init__(self, 
        df: pd.DataFrame,
        id1: str = 'id1',
        id2: str = 'id2',
        edge_weight: str = 'weight'
    ):
        node_list = pd.concat([df[id1], df[id2]]).unique() 
        edge_list = zip(df[id1], df[id2])
        edge_attr = {'edge_weight': df[edge_weight].values.tolist()}
        
        self.g = ig.Graph()
        self.g.add_vertices(node_list)
        self.g.add_edges(es=edge_list, attributes=edge_attr)
        
        
    def simple_partition(self, weight_attr: str = 'weight'):
        
        optimizer = la.Optimiser()
        pass
    
    
    def signed_partition(self, weight_attr: str = 'weight'):
        pass


    def merge_orgs(self, node_list, edge_list, edge_attr):
        pass


net_ppl = PeopleNet(df_edge_ppl, edge_weight='score_average_weighted')
# %%
