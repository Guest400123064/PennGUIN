# %%
from typing import List, FrozenSet, Tuple
import copy
import warnings

import pandas as pd
import numpy as np

import igraph as ig
import leidenalg as la
from pyvis.network import Network as VisNetwork


def draw_partition(g: ig.Graph, partition: List[FrozenSet[int]]) -> VisNetwork:
    """Helper function to generate HTML visualizations"""

    net = VisNetwork('1024px', '1024px')
    for i_node, i_grp in enumerate(partition):
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

    net.repulsion(central_gravity=0.1, spring_length=512)
    net.show_buttons(filter_=['nodes', 'edges', 'physics'])
    return net


# ======================================================================================================
# Load raw co-mention edge list and get a set of people (for plotting)
df_edge = pd.read_csv('../data/rw_edge_quote_regrade.csv')
ppl_set = set(
    pd.read_csv('../data/rw_ppl_list.csv', usecols=['persons'])
        .persons.str.lower()
        .unique()
        .tolist()
)

# Rename certain nodes (e.g., coref resolution)
rnm_map = lambda n: {
    'lady jeannette kagame': 'jeannette kagame',
    'paul kagame paulkagame': 'paul kagame',
    'diane shima rwigara': 'diane rwigara'
}.get(n, n)
df_edge.loc[:, 'id1'] = df_edge.loc[:, 'id1'].map(rnm_map)
df_edge.loc[:, 'id2'] = df_edge.loc[:, 'id2'].map(rnm_map)

# Drop nodes within blacklist
is_drop = lambda n: n in {
    'james karuhanga',
    'jean claude ntezimana',
    'peterson tumwebaze',
    'juvenal nkusi',
    'theoneste karenzi',
    'collins mwai',
    'ben gasore',
    'edmund kagire',
    'sharon kantengwa',
    'louise umutoni',
    'philippe mpayimana',
    'ignatius ssuuna',
    'gonza muganwa'
}
mask_drop = np.logical_or(df_edge.loc[:, 'id1'].map(is_drop), df_edge.loc[:, 'id2'].map(is_drop))
df_edge = df_edge.loc[~mask_drop]

# Drop self pointing edges
mask_loop = df_edge.loc[:, 'id1'] == df_edge.loc[:, 'id2']
df_edge = df_edge.loc[~mask_loop]

# Merge accross sentences
df_edge = (
    df_edge
        .dropna(subset='id2')
        .groupby(['id1', 'id2', 'flag_person', 'flag_company'], as_index=False, sort=False)
            [['regrade_score', 'co_mentions_count']]
        .sum()
)

# Calculate average tone and confidence of estimation (log10 count and normalize to [0, 1])
df_edge.loc[:, 'score_average']          = df_edge.regrade_score / df_edge.co_mentions_count
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
        
        # TODO: In the future we should NOT use string names!
        #   Because the names are NOT gauranteed to be UNIQUE;
        #     instead, we need to pre-fix a unique id for 
        #     each entity and use that as igraph node names.
        self.existing_names = set(node_list.tolist())
        
        
    def itos(self, idx: int) -> str:
        """Index to String. Given a node id, return the 
            node name, e.g., 42 -> Tony"""
        
        node = self.g.vs[idx]
        return node['name']
    
    
    def stoi(self, name: str) -> int:
        """String to Index. Given a node name, find the FIRST id 
            corresponds to that name, e.g., Tony -> 42"""
        
        found = self.g.vs.select(name=name)
        if len(found) == 1:
            return found[0].index
        elif len(found) > 1:
            warnings.warn(f'[WARN] :: duplicate name <{name}>; found {len(found)}')
        return -1
    
    
    def signed_partition(self, 
        resolution_pos: float = 0.01,
        resolution_neg: float = 0.3,
    ) -> List[int]:
        
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
        optimizer.set_rng_seed(42)
        optimizer.optimise_partition_multiplex(
            [part_pos, part_neg],
            layer_weights=[1, -1]
        )
        
        return part_pos.membership


    def signed_partition_merge_orgs(self, 
        node_list: List[str], 
        edge_list: List[Tuple[str, str, float]],
        ppl_membership: List[int],
        resolution_pos: float = 0.01,
        resolution_neg: float = 0.3
    ) -> Tuple[ig.Graph, List[int]]:
        
        # Merge org nodes & edges
        g_merge = copy.deepcopy(self.g)
        g_merge.add_vertices(list(set(node_list) - self.existing_names))
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
        optimizer.set_rng_seed(42)
        optimizer.optimise_partition_multiplex(
            [part_pos, part_neg],
            layer_weights=[1, -1],
            is_membership_fixed=fix_membership
        )
        
        return g_merge, part_pos.membership

# %%
# Clustering over people-co-mentions first
net_ppl = PeopleNetwork(df_edge_ppl, edge_weight='score_average_weighted')
mem_ppl = net_ppl.signed_partition(resolution_neg=0.02)

# Extend existing people-only graph with co-mentions involving companies
#   <g_merge> is the extended graph
#   <mem_org> is a list with same length as number of nodes; each element is the cluster id 
#       to which the corresponding node is assigned
node_list = pd.concat([df_edge_org['id1'], df_edge_org['id2']]).unique() 
edge_list = zip(df_edge_org['id1'], df_edge_org['id2'], df_edge_org['score_average_weighted'])
g_merge, mem_org = net_ppl.signed_partition_merge_orgs(
    node_list, edge_list, mem_ppl,
    resolution_pos=0.01,
    resolution_neg=0.3
)

# Generate html vis
net_vis = draw_partition(g_merge, mem_org)
for n in ppl_set:
    
    # Code peoples to squares and leave companies as dots
    found = g_merge.vs.select(name=n)
    if len(found) == 1:
        n = found[0].index
        net_vis.node_map[n]['shape'] = 'square'
    
net_vis.show('../out/rw_tone_cls_regrade.html')

# Write cluster assignments
pd.DataFrame({
    'entity_name': g_merge.vs['name'],
    'cluster': mem_org
}).to_csv('../out/rw_tone_cls_assignment_regrade.csv', index=False)

# %%
