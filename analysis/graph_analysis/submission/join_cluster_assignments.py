# %%
import pandas as pd


# Here I dropped singleton entities (no co-mention)
#   since the cluster assignments for them could be NaN
#   If the singletons are not-at-all co-mentioned, they 
#   will not participate graph-clustering
df_edges = pd.read_csv('rw_tone_merge.csv').dropna(subset='id2')
df_assign = pd.read_csv('rw_tone_cls_assignment.csv')

# Merge id1
df_join = pd.merge(
    df_edges, df_assign, 'left',
    left_on='id1', right_on='entity_name',
    suffixes=('', 'id1')
).rename({'cluster': 'id1_cluster'}, axis=1)

# Merge id2
df_join = pd.merge(
    df_join, df_assign, 'left',
    left_on='id2', right_on='entity_name',
    suffixes=('', 'id2')
).rename({'cluster': 'id2_cluster'}, axis=1)

# Dump file
df_join.to_csv('rw_tone_merge_cls_assignment.csv', index=False)
