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
from typing import List, FrozenSet
import sys
sys.path.append('./signed_local_community')

import pandas as pd
import numpy as np

from pyvis.network import Network

# Signed graph clustering
import networkx as nx
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

g = nx.from_pandas_edgelist(
    df=pd.DataFrame({
        'source': df_edge.loc[:, 'id1'],
        'target': df_edge.loc[:, 'id2'],
        'sign': df_edge.loc[:, 'co_mentions_avg'] * df_edge.loc[:, 'co_mentions_confidence'],
        'freq': df_edge.loc[:, 'co_mentions_count']
    }),
    edge_attr=['sign', 'freq']
)

stoi = {n: i for i, n in enumerate(g.nodes)}
itos = list(g.nodes)

# %%
s1 = [
    stoi['Paul Kagame']
]
s2 = [
    stoi['Frank Habineza']
]

x, obj_val = query_graph_using_sparse_linear_solver(
    g, [s1, s2], 
    kappa=0.9, 
    verbose=0
)

# Sweep on x to find C1 and C2
C1, C2, C, best_t, best_sbr, ts, sbr_list = sweep_on_x_fast(g, x, top_k=100)
for i, grp in enumerate([C1, C2]):
    for n in grp:
        g.nodes[itos[n]]['group'] = i
        g.nodes[itos[n]]['title'] = '[group] :: [{:}]'.format(i)

# %%
nx.set_edge_attributes(
    g, {(src, dst): {
        'width': abs(w),
        'color': 'blue' if w > 0 else 'red'
    } for (src, dst, w) in g.edges.data('sign')}
)

net = Network('1024px', '1024px', notebook=True)
net.from_nx(g)
net.show_buttons(filter_=['physics', 'edges'])
net.show('picture/cls-viz-tone-weight-by-freq-sign-local.html')

# %%
# Sanity check
opposition = [
    "Frank Habineza",
    "Paul Rusesabagina",
    "Kizito Mihigo",
    "Diane Rwigara"
]
cabinet = [
    "Vincent Biruta",
    "Louise Mushikiwabo",
    "Edouard Ngirente",
    "Bernard Makuza",
]
