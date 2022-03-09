
import pandas as pd


df_org = pd.read_csv('rw_tone_org.csv')
df_ppl = pd.read_csv('rw_tone_ppl.csv').assign(flag_person=1, flag_company=0)

df_ppl.loc[:, 'id1'] = df_ppl.loc[:, 'id1'].str.lower()
df_ppl.loc[:, 'id2'] = df_ppl.loc[:, 'id2'].str.lower()

df_out = (
    pd.concat([df_org, df_ppl])
        .reset_index()
        .drop_duplicates()
        .to_csv('rw_tone_merge.csv', index=False)
)
