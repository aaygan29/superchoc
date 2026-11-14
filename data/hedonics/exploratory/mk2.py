import pandas as pd, numpy as np
g=pd.read_csv('mixture_pleasantness_ma2021.csv')
b=pd.read_csv('raw/ma_2021/behavior.csv')
# panel level
pa,pb,ia,ib,y=g.pleas_A_alone,g.pleas_B_alone,g.int_A,g.int_B,g.pleasantness_mix
lo=np.minimum(pa,pb); hi=np.maximum(pa,pb)
print('n mixtures',len(g),'below min comp',(y<lo).mean(),'above max',(y>hi).mean(),'between',((y>=lo)&(y<=hi)).mean())
d=(pa-pb).abs(); print('abs diff comp pleas: median',d.median())
big=d>d.quantile(.5)
for name,m in [('contrast(top half |dP|)',big),('similar (bottom half)',~big)]:
    print(name,'mean y-mean',(y-(pa+pb)/2)[m].mean(),'y-lo',(y-lo)[m].mean(),'frac below both',(y[m]<lo[m]).mean(),'frac above both',(y[m]>hi[m]).mean())
# unpleasant dominance: y - mean vs lo
mean=(pa+pb)/2
wI=(pa*ia+pb*ib)/(ia+ib)
def rm(p): return np.sqrt(((p-y)**2).mean())
for n,p in [('mean',mean),('min',lo),('max',hi),('Iweighted',wI),('I2weighted',(pa*ia**2+pb*ib**2)/(ia**2+ib**2))]: print(n,'r=%.3f RMSE=%.3f'%(np.corrcoef(p,y)[0,1],rm(p)))
# linear regression
X=np.c_[np.ones(len(g)),wI,lo]; beta=np.linalg.lstsq(X,y,rcond=None)[0]; print('y~1+wI+lo',beta,'R2',1-((y-X@beta)**2).sum()/((y-y.mean())**2).sum())
X=np.c_[np.ones(len(g)),mean,lo]; beta=np.linalg.lstsq(X,y,rcond=None)[0]; print('y~1+mean+lo',beta)
# intensity of mixture vs components
print('int_mix - max(int)',(g.int_mix-np.maximum(ia,ib)).mean(),'int_mix - mean',(g.int_mix-(ia+ib)/2).mean(),' int_mix - (ia+ib)',(g.int_mix-(ia+ib)).mean())
# pleasantness vs mix intensity corr
print('corr(y, int_mix)',np.corrcoef(y,g.int_mix)[0,1])
# individual-level
b=b.dropna(); print('indiv corr PAB vs mean(PA,PB)',np.corrcoef(b.PAB,(b.PA+b.PB)/2)[0,1], 'vs min', np.corrcoef(b.PAB,np.minimum(b.PA,b.PB))[0,1])
# per-subject variation: split-half rater reliability on mixture PAB
s=b.groupby(['Stimulus A','Stimulus B','Subject']).PAB.mean().reset_index()
subs=sorted(s.Subject.unique()); rng=np.random.default_rng(0)
rs=[]
for _ in range(50):
    h=set(rng.choice(subs,len(subs)//2,replace=False)); a=s[s.Subject.isin(h)].groupby(['Stimulus A','Stimulus B']).PAB.mean(); c=s[~s.Subject.isin(h)].groupby(['Stimulus A','Stimulus B']).PAB.mean(); j=a.index.intersection(c.index); rs.append(np.corrcoef(a[j],c[j])[0,1])
print('split-half r (mean)',np.mean(rs), 'n subj', len(subs))
# duplicates exist? 
print('rep counts',b.groupby(['Stimulus A','Stimulus B','Subject']).size().value_counts().to_dict())
