"""
Paul Kagame (Tutsi, President)
Jeannette Kagame (First Lady)

Opposition politicians:
    Frank Habineza
    Paul Rusesabagina
    Kizito Mihigo
    Diane Rwigara

Kagame's cabinet:
    Vincent Biruta (current Minister of Foreign Affairs)
    Louise Mushikiwabo (previous Minister of Foreign Affairs)
    Edouard Ngirente (current PM)
    Bernard Makuza (Hutu, former PM)
"""

# %%
import readline
from typing import List, FrozenSet

import pandas as pd
import numpy as np

from pyvis.network import Network
import networkx as nx


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

g_str = nx.Graph()
g_str.add_weighted_edges_from(zip(
    df_edge.loc[:, 'id1'],
    df_edge.loc[:, 'id2'],
    df_edge.loc[:, 'co_mentions_avg'] * df_edge.loc[:, 'co_mentions_confidence']
))
g_int = nx.convert_node_labels_to_integers(g_str, label_attribute='label')
stoi = {n: i for i, n in enumerate(g_str.nodes)}
itos = list(g_str.nodes)

# %%
nx.write_weighted_edgelist(g_int, './test.txt')


# %%
def read_partition(path):
    
    with open(path) as f:
        for row in f.readlines():
            node, cluster = row.split()
            yield int(node), int(cluster)


assignment = read_partition('/home/lgfz1/Projects/pennguin/analysis/graph_analysis/signed_community_detection/partition.txt')
for node, cluster in assignment:
    g_int.nodes[node]['group'] = cluster
    g_int.nodes[node]['title'] = '[group] :: [{:}]'.format(cluster)

# %%
nx.set_edge_attributes(
    g_int, {(src, dst): {
        'width': abs(w),
        'color': 'blue' if w > 0 else 'red'
    } for (src, dst, w) in g_int.edges.data('weight')}
)

net = Network('1024px', '1280px', notebook=True)
net.from_nx(g_int)

# Use different shapes for cabinet and rivals
opposition = [
    "Frank Habineza",
    "Paul Rusesabagina",
    "Kizito Mihigo",
    "Diane Rwigara"
]
cabinet = [
    "Paul Kagame",
    "Vincent Biruta",
    "Louise Mushikiwabo",
    "Edouard Ngirente",
    "Bernard Makuza",
]
for o in opposition:
    net.node_map[stoi[o]]['shape'] = 'triangle'
    
for c in cabinet:
    net.node_map[stoi[c]]['shape'] = 'square'


# %%
net.show_buttons(filter_=['physics', 'edges'])
net.show('../picture/cls-viz-tone-weight-by-freq-sign-comm-detect.html')

# %%
