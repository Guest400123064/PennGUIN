# %%
from typing import List, FrozenSet, Tuple, Any, Dict, Union
import copy
import warnings

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
df_edge.loc[:, 'score_confidence']       = np.log(df_edge.co_mentions_count + 1)
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
        
        self.ppl_set = set(node_list.tolist())
        
        
    def itos(self, idx: int) -> str:
        
        node = self.g.vs[idx]
        return node['name']
    
    
    def stoi(self, name: str) -> int:
        
        found = self.g.vs.select(name=name)
        if len(found) == 1:
            return found[0].index
        elif len(found) > 1:
            warnings.warn(f'[WARN] :: duplicate name <{name}>; found {len(found)}')
        return -1
    
    
    def signed_partition(self, 
        resolution_pos: float = 0.01,
        resolution_neg: float = 0.3,
        return_member:  bool  = False,
        return_name:    bool  = False,
    ) -> Union[List[int], List[FrozenSet[int]], List[FrozenSet[str]]]:
        
        g_pos = self.g.subgraph_edges(self.g.es.select(edge_weight_gt=0), delete_vertices=False)
        g_neg = self.g.subgraph_edges(self.g.es.select(edge_weight_lt=0), delete_vertices=False)
        g_neg.es['edge_weight'] = [-w for w in g_neg.es['edge_weight']]
        
        part_pos = la.CPMVertexPartition(
            g_pos, 
            weights='edge_weight',
            resolution_parameter=resolution_pos
        )
        part_neg = la.CPMVertexPartition(
            g_neg, 
            weights='edge_weight',
            resolution_parameter=resolution_neg
        )
        
        optimizer = la.Optimiser()
        optimizer.optimise_partition_multiplex(
            [part_pos, part_neg],
            layer_weights=[1, -1]
        )
        
        if return_member:
            return part_pos.membership
        if return_name:
            return [frozenset(map(self.itos, c)) for c in part_pos]
        return [frozenset(c) for c in part_pos]
    

    def signed_partition_merge_orgs(self, 
        node_list: List[str], 
        edge_list: List[Tuple[str, str, float]],
        ppl_membership: List[int],
        return_name: bool     = False,
        resolution_pos: float = 0.01,
        resolution_neg: float = 0.3
    ) -> Tuple[ig.Graph, Union[List[FrozenSet[int]], List[FrozenSet[str]]]]:
        
        # Merge org nodes & edges
        g_merge = copy.deepcopy(self.g)
        g_merge.add_vertices(list(set(node_list) - self.ppl_set))
        for s, t, w in edge_list:
            g_merge.add_edge(s, t, edge_weight=w)
            
        # Find ppl membership first and fix
        fix_membership = [i < self.g.vcount() for i in range(g_merge.vcount())]
        new_membership = list(range(g_merge.vcount()))
        new_membership[:self.g.vcount()] = ppl_membership
        
        # Merge partitions for organizations
        g_pos = g_merge.subgraph_edges(g_merge.es.select(edge_weight_gt=0), delete_vertices=False)
        g_neg = g_merge.subgraph_edges(g_merge.es.select(edge_weight_lt=0), delete_vertices=False)
        g_neg.es['edge_weight'] = [-w for w in g_neg.es['edge_weight']]
        
        part_pos = la.CPMVertexPartition(
            g_pos, new_membership,
            weights='edge_weight',
            resolution_parameter=resolution_pos
        )
        part_neg = la.CPMVertexPartition(
            g_neg, new_membership,
            weights='edge_weight',
            resolution_parameter=resolution_neg
        )
        
        optimizer = la.Optimiser()
        optimizer.optimise_partition_multiplex(
            [part_pos, part_neg],
            layer_weights=[1, -1],
            is_membership_fixed=fix_membership
        )
        
        if return_name:
            return g_merge, [frozenset(map(self.itos, c)) for c in part_pos]
        return g_merge, [frozenset(c) for c in part_pos]


net_ppl = PeopleNetwork(df_edge_ppl, edge_weight='score_average')
mem_ppl = net_ppl.signed_partition(return_member=True)

# %%
node_list = pd.concat([df_edge_org['id1'], df_edge_org['id2']]).unique() 
edge_list = zip(df_edge_org['id1'], df_edge_org['id2'], df_edge_org['score_average'])

g_merge, partition = net_ppl.signed_partition_merge_orgs(node_list, edge_list, mem_ppl)
draw_partition(g_merge, partition).show('tmp.html')

# %%
