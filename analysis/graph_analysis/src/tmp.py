# %%
import pandas as pd


df_ppl_org = pd.read_csv('/home/lgfz1/Projects/pennguin/analysis/graph_analysis/data/rw_top_200_tone_w_org.csv')
df_ppl = pd.read_csv('/home/lgfz1/Projects/pennguin/analysis/graph_analysis/data/rw_top_200_tone.csv').assign(flag_person=1, flag_company=0)
df_ppl.loc[:, 'id1'] = df_ppl.loc[:, 'id1'].str.lower()
df_ppl.loc[:, 'id2'] = df_ppl.loc[:, 'id2'].str.lower()
df_out = (
    pd.concat([df_ppl_org, df_ppl])
        .reset_index()
        .drop_duplicates()
        .to_csv('/home/lgfz1/Projects/pennguin/analysis/graph_analysis/data/rw_top_200_tone_w_org_merge.csv', index=False)
)

# %%
