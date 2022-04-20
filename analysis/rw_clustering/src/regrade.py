# %%
import sys
sys.path.append('../../..')

import pandas as pd
import json

# Prepare event extractor and grader
from pennguin import KeyBERTEventExtractor, GoldsteinGrader
with open('../model/goldstein.json') as f:
    goldstein = json.load(f)
    extractor = KeyBERTEventExtractor('all-mpnet-base-v2')
    grader = GoldsteinGrader(goldstein, extractor)

# %%
# Regrade all co-mention sentences
df_edge = pd.read_csv('../data/rw_edge_quote.csv')
df_edge.loc[:, 'regrade_score'] = grader.grade(df_edge.sentence.values.tolist())
df_edge.to_csv('../data/rw_edge_quote_regrade.csv', index=False)
