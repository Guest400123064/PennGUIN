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

import pandas as pd
import numpy as np

from pyvis.network import Network

# Signed graph clustering
import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities


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
        'prob': np.exp(df_edge.loc[:, 'co_mentions_avg'] * df_edge.loc[:, 'co_mentions_confidence']),
        'tone': df_edge.loc[:, 'co_mentions_avg'] * df_edge.loc[:, 'co_mentions_confidence'],
        'freq': df_edge.loc[:, 'co_mentions_count']
    }),
    edge_attr=['prob', 'tone', 'freq']
)

stoi = {n: i for i, n in enumerate(g.nodes)}
itos = list(g.nodes)

# %%
# Compute partition
groups = asyn_lpa_communities(g, weight='prob')

# Assign labels
for i, grp in enumerate(groups):
    for n in grp:
        g.nodes[n]['group'] = i
        g.nodes[n]['title'] = '[group] :: [{:}]'.format(i)

# %%
nx.set_edge_attributes(
    g, {(src, dst): {
        'width': abs(w),
        'color': 'blue' if w > 0 else 'red'
    } for (src, dst, w) in g.edges.data('tone')}
)

net = Network('1024px', '1024px', notebook=True)
net.from_nx(g)
net.show_buttons(filter_=['physics', 'edges'])
net.show('picture/cls-viz-tone-weight-by-freq-lpa.html')

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
