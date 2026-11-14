import pandas as pd
b=pd.read_csv('raw/ma_2021/behavior.csv'); s=pd.read_csv('raw/ma_2021/stimuli.csv'); m=pd.read_csv('raw/ma_2021/molecules.csv')
n=m.set_index('CID').name; S=s.set_index('Stimulus')
b['mixture_id']=['ma2021_%d_%d'%(a,c) for a,c in zip(b['Stimulus A'],b['Stimulus B'])]
b['components']=[n[S.loc[a,'CID']]+';'+n[S.loc[c,'CID']] for a,c in zip(b['Stimulus A'],b['Stimulus B'])]
b['cids']=[f"{S.loc[a,'CID']};{S.loc[c,'CID']}" for a,c in zip(b['Stimulus A'],b['Stimulus B'])]
b['conc_mg_per_mL']=[f"{S.loc[a,'Concentration (mg/mL)']};{S.loc[c,'Concentration (mg/mL)']}" for a,c in zip(b['Stimulus A'],b['Stimulus B'])]
o=b.rename(columns={'Subject':'rater','Rep':'rep','PAB':'pleasantness_mix','PA':'pleas_A_alone','PB':'pleas_B_alone','IAB':'int_mix','IA':'int_A','IB':'int_B'})
o['scale']='0-10 VAS (100 mm; pleasantness endpoints not labelled on data-in-brief page)'; o['source_key']='ma_2021'
o[['mixture_id','components','cids','conc_mg_per_mL','rater','rep','pleasantness_mix','pleas_A_alone','pleas_B_alone','int_A','int_B','int_mix','scale','source_key']].to_csv('mixture_pleasantness_ma2021_long.csv',index=False)
print(len(o))
