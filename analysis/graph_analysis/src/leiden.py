# %%
from typing import List, FrozenSet
from collections import defaultdict

import pandas as pd
import numpy as np

from pyvis.network import Network
import networkx as nx
import igraph as ig
import leidenalg as la


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


# Create weight graph
g = ig.Graph()
g.add_vertices(
    pd.concat([df_edge.loc[:, 'id1'], df_edge.loc[:, 'id2']]).unique()
)
g.add_edges(
    es=zip(df_edge.loc[:, 'id1'], df_edge.loc[:, 'id2']),
    attributes={'weight': df_edge.loc[:, 'co_mentions_avg'] * df_edge.loc[:, 'co_mentions_confidence']}
)

# %%
# Partition to positive and negative graphs and do separate partition
g_pos = g.subgraph_edges(g.es.select(weight_gt=0), delete_vertices=False)
g_neg = g.subgraph_edges(g.es.select(weight_lt=0), delete_vertices=False)
g_neg.es['weight'] = [-w for w in g_neg.es['weight']]

part_pos = la.CPMVertexPartition(g_pos, weights='weight', resolution_parameter=0.0)
part_neg = la.CPMVertexPartition(g_neg, weights='weight', resolution_parameter=0.0)

optimizer = la.Optimiser()
diff = optimizer.optimise_partition_multiplex(
    [part_pos, part_neg], layer_weights=[1, -1]
)
print(list(part_pos))

# %%
net = Network('1024px', '1280px', notebook=True)
