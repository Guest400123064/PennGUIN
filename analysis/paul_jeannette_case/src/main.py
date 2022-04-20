# %%
import sys
sys.path.append('../../..')

import numpy as np
import pandas as pd

import json
from pprint import pprint

# Prepare event extractor and grader
from pennguin import KeyBERTEventExtractor, GoldsteinGrader
with open('../model/goldstein.json') as f:
    goldstein = json.load(f)
    event_lst = list(goldstein.keys())
    extractor = KeyBERTEventExtractor('all-mpnet-base-v2')
    grader = GoldsteinGrader(goldstein, extractor)

# %%
# Read co-mentions of Paul and Jeannette
#   - 1. lower all ids
#   - 2. rename
#   - 3. filtering
df_edge = pd.read_csv('../data/rw_edge_quote.csv')
df_edge.loc[:, 'id1'] = df_edge.loc[:, 'id1'].str.lower()
df_edge.loc[:, 'id2'] = df_edge.loc[:, 'id2'].str.lower()

rnm_map = lambda n: {
    'lady jeannette kagame': 'jeannette kagame',
    'paul kagame paulkagame': 'paul kagame',
}.get(n, n)
df_edge.loc[:, 'id1'] = df_edge.loc[:, 'id1'].map(rnm_map)
df_edge.loc[:, 'id2'] = df_edge.loc[:, 'id2'].map(rnm_map)

mask_jean = np.logical_or(
    df_edge.loc[:, 'id1'] == 'jeannette kagame',
    df_edge.loc[:, 'id2'] == 'jeannette kagame'
)
mask_paul = np.logical_or(
    df_edge.loc[:, 'id1'] == 'paul kagame',
    df_edge.loc[:, 'id2'] == 'paul kagame'
)
df_edge = df_edge.loc[
    np.logical_and(mask_jean, mask_paul), 
    ['article', 'id1', 'id2', 'avg1', 'sentence']
]

# %%
# Case 1: 
#   - low confidence score (cosine) results in inaccurate grading
# Solution:
#   - larger language model, e.g., 'all-mpnet-base-v2', performs significantly better.
#       Especially the models that are trained upon NLI tasks.
txt_low_conf = [
    'Xi who was flanked by his wife Peng Liyuan was received yesterday by President Paul Kagame and his wife his wife Jeannette Kagame.',
    'Nation World. Rwanda somberly marks the start of genocide 25 years ago. President Paul Kagame and first lady Jeannette Kagame laid wreaths and lit a flame at the mass burial ground of 250,000 victims at the Kigali Genocide Memorial Center in the capital, Kigali.',
    '\nThere is broad agreement that this success is attributable in large measure to the strong leadership and dedication of President Paul Kagame and the First Lady of Rwanda, Mrs Jeannette Kagame through her Imbuto Foundation, all underpinned by robust and innovative Government programmes, broad involvement of all the stakeholders and communities as well as adequate external support.'
]
ext_low_conf = extractor.extract(txt_low_conf, event_lst)
pprint(ext_low_conf)

# %%
# Case 2: 
#   - wrong event-comention association; the entities simply 
#       appears together with the event but without any causal relationship 
txt_mis_link = [
    'At least 800,000 Tutsi and moderate Hutus murdered during three-month genocide\n            \n\n\n\nSun, Apr 7, 2019, 13:15\nUpdated: Sun, Apr 7, 2019, 13:21\n\n\n\n \n  \n\n  \n\nPresident of Rwanda Paul Kagame (L) and his wife and first lady of Rwanda Jeannette Kagame (R) arrive for a commemoration event.'
]
ext_mis_link = extractor.extract(txt_mis_link, event_lst)
pprint(ext_mis_link)

# %%
# Case 3: 
#   - mis-classification with relatively high confidence score
# Solution:
#   - larger language model, e.g., 'all-mpnet-base-v2', performs significantly better.
#       Especially the models that are trained upon NLI tasks.
txt_wrong_high_conf = [
    'President Xi was welcomed by his Rwandan counterpart, President Paul Kagame, his wife Jeannette Kagame.'
]
ext_wrong_high_conf = extractor.extract(txt_wrong_high_conf, event_lst)
pprint(ext_wrong_high_conf)

# %%
# Case 4*: 
#   - noisy text, containing many irrelevant characters; though grading performance is not 
#       substantially affected.
txt_noisy = [
    'SportsLifestyleWellnessPeopleEntertainmentSocietyWeekenderVideoJobs & TendersEpaperPodcast\n\n\n\n\n\n\nTwitter\nFacebook\nEmail\n\n\n\n\n\n\n\n\n\n\n\nSearch form\n\nSearch\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nNews\nRwanda, Djibouti sign five bilateral agreements\n\nRwanda and Djibouti yesterday signed five bilateral agreements during the first day of a two-day state visit by President Paul Kagame and First Lady Jeannette Kagame to Djibouti.\n\n\n'
]
ext_noisy = extractor.extract(txt_noisy, event_lst)
pprint(ext_noisy)

# %%
# Case 5* duplicate sentences. 
# Potential causes: 
#   - Different name for same entity, e.g., JK & Lady JK
#   - Duplicate articles, there are duplicate sentences across different article ids.
#       This might be caused by same article appearing on different websites, 
#       e.g., gkg-id 20190407154500-847
# Example ids are
#   - (5808, 5809)
#   - (1067, 4414)
#   - (1348, 1350, 2608, 3039, 3099, 4582, 4584)
