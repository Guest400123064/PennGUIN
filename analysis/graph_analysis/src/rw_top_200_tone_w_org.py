# %%
from typing import List, FrozenSet, Tuple, Any, Dict, Union

import pandas as pd
import numpy as np

import igraph as ig
import leidenalg as la
from pyvis.network import Network as VisNetwork


def draw_partition(
    g: ig.Graph, 
    partition: List[FrozenSet[int]]
) -> VisNetwork:

    net = VisNetwork('1024px', '1024px')
    for i_grp, grp in enumerate(partition):
        for i_node in grp:
            net.add_node(
                i_node, 
                value=2,
                label=g.vs[i_node]['name'],
                group=i_grp,
                title=f'Group {i_grp}'
            )
            
    for edge in g.es:
        net.add_edge(
            edge.source, edge.target,
            title=round(edge['edge_weight'], 2),
            value=abs(edge['edge_weight']),
            color='red' if (edge['edge_weight'] < 0) else 'blue'
        )

    net.repulsion(central_gravity=0.1)
    net.show_buttons(filter_=['physics', 'edges'])
    return net


# ======================================================================================================
# Load raw co-mention edge list and merge accross articles
#   IMPORTANT: Though there are fields <co_mentions_sum> 
#       and <co_mentions_count> in the source edge list file, 
#       they are PER ARTICLE statistics. So we still need 
#       to sum them up individually and manually calculate average
df_edge = df_edge = (
    pd.read_csv('/home/lgfz1/Projects/pennguin/analysis/graph_analysis/data/rw_top_200_tone_w_org.csv')
        .dropna(subset='id2')
        .groupby(['id1', 'id2', 'flag_person', 'flag_company'], as_index=False, sort=False)
            [['co_mentions_sum', 'co_mentions_count']]
        .sum()
)

# Calculate average tone and confidence of estimation (log10 count and normalize to [0, 1])
df_edge.loc[:, 'score_average']          = df_edge.co_mentions_sum / df_edge.co_mentions_count
df_edge.loc[:, 'score_confidence']       = np.log10(df_edge.co_mentions_count + 1)
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
class PeopleNetwork:
    
    def __init__(self, 
        df: pd.DataFrame,
        id1: str         = 'id1',
        id2: str         = 'id2',
        edge_weight: str = 'weight'
    ) -> None:
        node_list = pd.concat([df[id1], df[id2]]).unique() 
        edge_list = zip(df[id1], df[id2])
        edge_attr = {'edge_weight': df[edge_weight].values.tolist()}
        
        self.g = ig.Graph()
        self.g.add_vertices(node_list)
        self.g.add_edges(es=edge_list, attributes=edge_attr)
        
        
    def itos(self, idx: int) -> str:
        
        node = self.g.vs[idx]
        return node['name']
    
    
    def stoi(self, name: str) -> int:
        
        names = self.g.vs['name']
        return names.index(name)
        
        
    def simple_partition(self, 
        return_name: bool = False,
        resolution: float = 0.01
    ) -> Union[List[FrozenSet[int]], List[FrozenSet[str]]]:
        
        partition = la.find_partition(
            self.g, 
            partition_type       = la.CPMVertexPartition,
            resolution_parameter = resolution,
            weights              = 'edge_weight',
            n_iterations         = 8,
            seed                 = 42
        )
        
        if return_name:
            return [frozenset(map(self.itos, c)) for c in partition]
        return [frozenset(c) for c in partition]
    
    
    def signed_partition(self, 
        return_name: bool     = False,
        resolution_pos: float = 0.01,
        resolution_neg: float = 0.3
    ) -> Union[List[FrozenSet[int]], List[FrozenSet[str]]]:
        
        g_pos = self.g.subgraph_edges(self.g.es.select(edge_weight_gt=0), delete_vertices=False)
        g_neg = self.g.subgraph_edges(self.g.es.select(edge_weight_lt=0), delete_vertices=False)
        g_neg.es['edge_weight'] = [-w for w in g_neg.es['edge_weight']]
        
        part_pos = la.CPMVertexPartition(
            g_pos, 
            weights='edge_weight',
            resolution_parameter = resolution_pos
        )
        part_neg = la.CPMVertexPartition(
            g_neg, 
            weights='edge_weight',
            resolution_parameter = resolution_neg
        )
        
        optimizer = la.Optimiser()
        optimizer.optimise_partition_multiplex(
            [part_pos, part_neg],
            layer_weights=[1, -1]
        )
        
        if return_name:
            return [frozenset(map(self.itos, c)) for c in part_pos]
        return [frozenset(c) for c in part_pos]
    

    def merge_orgs(self, node_list, edge_list, edge_attr):
        pass


net_ppl = PeopleNetwork(df_edge_ppl, edge_weight='score_average')

# %%
partition = net_ppl.signed_partition(resolution_neg=0.3)
draw_partition(net_ppl.g, partition).show('tmp.html')

# %%
