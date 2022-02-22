# %%
from typing import List, FrozenSet
import sys
sys.path.append('./signed_local_community')

import pandas as pd
import numpy as np

from pyvis.network import Network

import networkx as nx
from networkx.algorithms.community import (
    k_clique_communities,
    greedy_modularity_communities
)

# Signed graph clustering
from signed_local_community.core import (
    query_graph_using_sparse_linear_solver, 
    sweep_on_x_fast
)


# Read Tone-CoMention results
df_result = pd.read_csv(r'/home/lgfz1/Projects/pennguin/analysis/graph_analysis/data/rw_top_200_tone.csv')
df_result = df_result[~df_result.id2.isna()]

df_edge = (df_result
    .groupby(['id1', 'id2'])[['co_mentions_sum', 'co_mentions_count']]
    .sum()
    .reset_index())
df_edge.loc[:, 'co_mentions_avg'] = df_edge.co_mentions_sum / df_edge.co_mentions_count
df_edge.loc[:, 'co_mentions_confidence'] = np.log10(df_edge.co_mentions_count + 1)
df_edge.loc[:, 'co_mentions_confidence'] = df_edge.co_mentions_confidence / df_edge.co_mentions_confidence.max()

# %%
# Build an Entity-to-NodeIndex table
class Entities:
    
    def __init__(self, G: nx.Graph) -> None:
        self._g = G

    def __len__(self) -> int:
        return len(self.G)

    @property
    def G(self) -> nx.Graph:
        return self._g
    
    @property
    def cliques(self) -> List[FrozenSet[str]]:
        return sorted(
            map(frozenset, nx.find_cliques(self.G)), 
            reverse=True, key=len
        )

    def cls_k_clique(self, k: int) -> List[FrozenSet[str]]:
        
        # Compute groups
        groups = list(k_clique_communities(self.G, k))
        others = set(self.G.nodes.keys())
        
        # Assign labels
        for i, g in enumerate(groups):
            for n in g:
                others -= g
                self.G.nodes[n]['group'] = i
                self.G.nodes[n]['title'] = '[group] :: [{:}]'.format(i)

        # All other nodes in a default group
        for n in others:
            self.G.nodes[n]['group'] = len(groups)
            self.G.nodes[n]['title'] = '[group] :: [DEFAULT]'
        return groups + [frozenset(others)]
    
    def cls_greedy_modularity(self, weights: str = None) -> List[FrozenSet[str]]:
        
        # Compute partition
        groups = greedy_modularity_communities(self.G, weight=weights)

        # Assign labels
        for i, g in enumerate(groups):
            for n in g:
                self.G.nodes[n]['group'] = i
                self.G.nodes[n]['title'] = '[group] :: [{:}]'.format(i)

        return groups


# Init ent table and add edges
ent = Entities(
    nx.from_pandas_edgelist(
        df=pd.DataFrame({
            'source': df_edge.loc[:, 'id1'],
            'target': df_edge.loc[:, 'id2'],
            'tone_raw': df_edge.loc[:, 'co_mentions_avg'],
            'tone_pos': 10 + df_edge.loc[:, 'co_mentions_avg'],
            'tone_neg': 10 - df_edge.loc[:, 'co_mentions_avg'],
            'tone_pos_weight': 10 + df_edge.loc[:, 'co_mentions_avg'] * df_edge.loc[:, 'co_mentions_confidence'],
            'freq': df_edge.loc[:, 'co_mentions_count']
        }),
        edge_attr=['tone_raw', 'tone_pos', 'tone_neg', 'freq']
    )
)

# %%
cls_percolation = ent.cls_k_clique(5)
cls_modularity = ent.cls_greedy_modularity('tone_pos_weight')

# %%
# Visualization
nx.set_edge_attributes(
    ent.G, {(src, dst): {
        'width': abs(w - 10),
        'color': 'blue' if w > 10 else 'red'
    } for (src, dst, w) in ent.G.edges.data('tone_pos')}
)

net = Network('1024px', '1024px', notebook=True)
net.from_nx(ent.G)
net.show_buttons(filter_=['physics', 'edges'])
net.show('picture/cls-viz-tone-weight-by-freq.html')

# %%
def cls2df(clusters: List[FrozenSet[str]]) -> pd.DataFrame:
    
    ret = []
    for c, s in enumerate(clusters):
        ret.extend(zip(s, [c] * len(s)))

    df = pd.DataFrame(ret, columns=['name', 'cluster'])
    return df.groupby('name').first()

# %%
