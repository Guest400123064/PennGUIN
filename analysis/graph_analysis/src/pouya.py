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
from collections import defaultdict
import subprocess
import os

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

g = nx.Graph()
g.add_weighted_edges_from(zip(
    df_edge.loc[:, 'id1'],
    df_edge.loc[:, 'id2'],
    df_edge.loc[:, 'co_mentions_avg'] * df_edge.loc[:, 'co_mentions_confidence']
))

# %%
def partition(
    g: nx.Graph, 
    r: float = 1,
    exe_path: str = '../signed_community_detection',
    exe_name: str = 'main.jar'
) -> List[FrozenSet]:
    
    itos = list(g.nodes)
    
    # Make tmp dir for computing partition
    tmp_dir = os.path.join(exe_path, 'tmp')
    graph_path = os.path.join(tmp_dir, 'g.txt')
    partition_path = os.path.join(tmp_dir, 'p.txt')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    # Export integer labeled graph
    g_int = nx.convert_node_labels_to_integers(g, label_attribute='label')
    nx.write_weighted_edgelist(g_int, graph_path)

    # Comput partition
    exe = os.path.join(exe_path, exe_name)
    cmd = ('java ' + 
            f'-jar {exe} mdl --verbose ' + 
            f'-g   {graph_path} ' + 
            f'-o   {partition_path} ' + 
            f'-r   {r}')
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        raise Exception('Error: cannot run partition executable')
    
    # Load partition results
    ret = defaultdict(set)
    with open(partition_path) as f:
        for row in f.readlines():
            node, cluster = row.split()
            ret[int(cluster)].add(itos[int(node)])
            
    return [frozenset(s) for s in ret.values()]
    

assignments = partition(g)
for i, grp in enumerate(assignments):
    for node in grp:
        g.nodes[node]['group'] = i
        g.nodes[node]['title'] = f'[group] :: [{i}]'

# %%
nx.set_edge_attributes(
    g, {(src, dst): {
        'width': abs(w),
        'color': 'blue' if w > 0 else 'red'
    } for (src, dst, w) in g.edges.data('weight')}
)

net = Network('1024px', '1280px', notebook=True)
net.from_nx(g)

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
    net.node_map[o]['shape'] = 'triangle'
    
for c in cabinet:
    net.node_map[c]['shape'] = 'square'

# Draw
net.show_buttons(filter_=['physics', 'edges'])
net.show('../picture/cls-viz-tone-weight-by-freq-sign-comm-detect.html')

# %%
