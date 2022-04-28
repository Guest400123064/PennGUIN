# %%
from typing import List, FrozenSet, Tuple
import copy
import warnings

import pandas as pd
import numpy as np

import igraph as ig
import leidenalg as la
from pyvis.network import Network as VisNetwork


def draw_partition(g: ig.Graph, partition: List[FrozenSet[int]], cutoff: float = 0) -> VisNetwork:
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
            color='red' if (edge['edge_weight'] < cutoff) else 'blue'
        )

    net.repulsion(central_gravity=0.1, spring_length=512)
    net.show_buttons(filter_=['nodes', 'edges', 'physics'])
    return net


# ======================================================================================================
# Load raw co-mention edge list and get a set of people and org (for plotting)
df_edge = pd.read_csv('../data/ni_delta_edge_list.csv')
df_edge.loc[:, 'entity1'] = df_edge.loc[:, 'entity1'].str.title()
df_edge.loc[:, 'entity2'] = df_edge.loc[:, 'entity2'].str.title()

mask_ppl = np.logical_and(
    df_edge.loc[:, 'person_flag'] == 1, 
    df_edge.loc[:, 'org_flag'] == 0
)
mask_org = np.logical_and(
    df_edge.loc[:, 'person_flag'] == 0, 
    df_edge.loc[:, 'org_flag'] == 1
)
ppl_set = set(
    df_edge.loc[mask_ppl, 'entity1'].str.title()
).union(set(
    df_edge.loc[mask_ppl, 'entity2'].str.title()
))
org_set = set(
    df_edge.loc[mask_org, 'entity1'].str.title()
).union(set(
    df_edge.loc[mask_org, 'entity2'].str.title()
))

# Merge accross articles
df_edge = (
    df_edge
        .dropna(subset='entity2')
        .groupby(['entity1', 'entity2', 'person_flag', 'org_flag'], as_index=False, sort=False)
            [['tone_sum', 'co_mention_count']]
        .sum()
)

# Filter low number of co-mention, e.g., < 10
df_edge = df_edge.loc[df_edge.co_mention_count >= 1000].reset_index(drop=True)

# Calculate average tone and confidence of estimation (log10 count and normalize to [0, 1])
df_edge.loc[:, 'score_average']          = df_edge.tone_sum / df_edge.co_mention_count
df_edge.loc[:, 'score_confidence']       = np.log(df_edge.co_mention_count)
df_edge.loc[:, 'score_confidence']       = df_edge.score_confidence / df_edge.score_confidence.max()
df_edge.loc[:, 'score_average_weighted'] = df_edge.score_average * df_edge.score_confidence


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
# Ego network
seed_ent = 'Exxon' # 'Chevron', 'Shell', 'Exxon'
ego_ent_set = {seed_ent}

# Get one-hop entities
mask_ego = np.logical_or(
    df_edge.entity1.map(lambda x: x in ego_ent_set),
    df_edge.entity2.map(lambda x: x in ego_ent_set)
)
ego_ent_set = ego_ent_set.union(set(
    df_edge.loc[mask_ego, 'entity1'].str.title()
).union(set(
    df_edge.loc[mask_ego, 'entity2'].str.title()
)))

# Get two-hop entities
mask_ego = np.logical_and(
    df_edge.entity1.map(lambda x: x in ego_ent_set),
    df_edge.entity2.map(lambda x: x in ego_ent_set)
)

# People only visualization
net_ego = PeopleNetwork(df_edge[mask_ego], 'entity1', 'entity2', edge_weight='score_average_weighted')
mem_ego = net_ego.signed_partition(resolution_neg=0.02)

# Generate html vis
net_vis = draw_partition(net_ego.g, mem_ego, -1)
for n in ppl_set:
    
    # Code peoples to squares and leave companies as dots
    found = net_ego.g.vs.select(name=n)
    if len(found) == 1:
        n = found[0].index
        net_vis.node_map[n]['shape'] = 'square'

net_vis.show(f'../out/ni_tone_cls_{seed_ent}.html')

# Write cluster assignments
pd.DataFrame({
    'entity_name': net_ego.g.vs['name'],
    'cluster': mem_ego
}).to_csv(f'../out/ni_tone_cls_{seed_ent}_assign.csv', index=False)

# %%
# Filter out people-only edges
df_edge_ppl = df_edge.loc[mask_ppl]
df_edge_org = df_edge.loc[mask_org]
df_edge_mix = df_edge.loc[~mask_ppl]

# Get top ppl
df_top_ppl = pd.read_csv('../data/ni_top_ppl.csv')
top_ppl_set = set(df_top_ppl.sort_values('n', ascending=False).persons.head(50).values.tolist())
mask_top_ppl = np.logical_and(
    df_edge_ppl.entity1.map(lambda x: x in top_ppl_set),
    df_edge_ppl.entity2.map(lambda x: x in top_ppl_set)
)

# Get top org
df_top_org = pd.read_csv('../data/ni_top_orgs.csv')
top_org_set = set(df_top_org.sort_values('n', ascending=False).organizations.head(50).values.tolist())
mask_top_org = np.logical_and(
    df_edge_org.entity1.map(lambda x: x in top_org_set),
    df_edge_org.entity2.map(lambda x: x in top_org_set)
)

# %%
# People only visualization
net_ppl = PeopleNetwork(df_edge_ppl[mask_top_ppl], 'entity1', 'entity2', edge_weight='score_average_weighted')
mem_ppl = net_ppl.signed_partition(resolution_neg=0.02)

# Generate html vis
net_vis = draw_partition(net_ppl.g, mem_ppl)
net_vis.show('../out/ni_tone_cls_ppl.html')

# Write cluster assignments
pd.DataFrame({
    'entity_name': net_ppl.g.vs['name'],
    'cluster': mem_ppl
}).to_csv('../out/ni_tone_cls_ppl_assign.csv', index=False)

# %%
# Org only visualization
net_org = PeopleNetwork(df_edge_org[mask_top_org], 'entity1', 'entity2', edge_weight='score_average_weighted')
mem_org = net_org.signed_partition(resolution_neg=0.02)

# Generate html vis
net_vis = draw_partition(net_org.g, mem_org)
net_vis.show('../out/ni_tone_cls_org.html')

# Write cluster assignments
pd.DataFrame({
    'entity_name': net_org.g.vs['name'],
    'cluster': mem_org
}).to_csv('../out/ni_tone_cls_org_assign.csv', index=False)

# %%
# Mixed visualization

# Top mixture
top_ppl_set = set(df_top_ppl.sort_values('n', ascending=False).persons.head(25).values.tolist())
top_org_set = set(df_top_org.sort_values('n', ascending=False).organizations.head(25).values.tolist())

mask_top_ppl = np.logical_and(
    df_edge_ppl.entity1.map(lambda x: x in top_ppl_set),
    df_edge_ppl.entity2.map(lambda x: x in top_ppl_set)
)

mask_top_mix = np.logical_and(
    np.logical_or(
        df_edge_mix.entity1.map(lambda x: x in top_ppl_set),
        df_edge_mix.entity1.map(lambda x: x in top_org_set)
    ),
    np.logical_or(
        df_edge_mix.entity2.map(lambda x: x in top_ppl_set),
        df_edge_mix.entity2.map(lambda x: x in top_org_set)
    )
)
df_edge_mix = df_edge_mix[mask_top_mix]

# Clustering over people-co-mentions first
net_ppl = PeopleNetwork(df_edge_ppl[mask_top_ppl], 'entity1', 'entity2', edge_weight='score_average_weighted')
mem_ppl = net_ppl.signed_partition(resolution_neg=0.02)

# Extend existing people-only graph with co-mentions involving companies
#   <g_merge> is the extended graph
#   <mem_org> is a list with same length as number of nodes; each element is the cluster id 
#       to which the corresponding node is assigned
node_list = pd.concat([df_edge_mix['entity1'], df_edge_mix['entity2']]).unique() 
edge_list = zip(df_edge_mix['entity1'], df_edge_mix['entity2'], df_edge_mix['score_average_weighted'])
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
    
net_vis.show('../out/ni_tone_cls_ppl_org.html')

# Write cluster assignments
pd.DataFrame({
    'entity_name': g_merge.vs['name'],
    'cluster': mem_org
}).to_csv('../out/ni_tone_cls_ppl_org_assign.csv', index=False)

# %%
