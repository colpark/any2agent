"""Local-only, confidence-aware distillation of noisy land-cover evidence."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import json
import re
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

threadpool_limits(2)
ROOT = Path(__file__).resolve().parent
classes = json.loads((ROOT / 'classes.json').read_text())
ids = json.loads((ROOT / 'test_ids.json').read_text())
keys = [str(c['id']) for c in classes]

def read_rows(name, n):
    rows = [json.loads(s) for s in (ROOT / name).read_text().splitlines()]
    assert len(rows) == n and sorted(x['row'] for x in rows) == list(range(n))
    return sorted(rows, key=lambda x: x['row'])

F = np.vstack([np.load(ROOT / 'bank_features.npy'), np.load(ROOT / 'test_features.npy')]).astype(float)
D = np.vstack([np.load(ROOT / 'bank_descriptors.npy'), np.load(ROOT / 'test_descriptors.npy')]).astype(float)
n = len(F) - len(ids)
nt = len(ids)
assert F.shape == (n+nt,4096) and D.shape == (n+nt,10)
Qrows = read_rows('bank_qwen.jsonl',n) + read_rows('test_qwen.jsonl',nt)
Arows = read_rows('bank_astra.jsonl',n)
Crows = read_rows('bank_captions.jsonl',n) + read_rows('test_captions.jsonl',nt)

conf = {'high': 1., 'medium': .65, 'low': .3}
Q = np.array([[((1 if r['answers'][k]['answer']=='yes' else -1)*conf[r['answers'][k]['confidence']]) for k in keys] for r in Qrows])
Qcat = np.array([[{'no':0,'yes':2}[r['answers'][k]['answer']] + (r['answers'][k]['confidence']=='high') for k in keys] for r in Qrows])
hard = np.array([[r['answers'][k]['answer']=='yes' for k in keys] for r in Arows],dtype=float)
mapping = {'yes': {'high': .97, 'medium': .80, 'low': .57}, 'no': {'high': .015, 'medium': .10, 'low': .28}}
Y = np.array([[mapping[r['answers'][k]['answer']][r['answers'][k]['confidence']] for k in keys] for r in Arows])

def string_values(obj):
    if isinstance(obj,str):
        return obj.replace('_',' ')
    if isinstance(obj,list):
        return ' '.join(string_values(x) for x in obj)
    if isinstance(obj,dict):
        return ' '.join(string_values(v) for k,v in obj.items() if k not in ['cell','confidence','not_revealed','uncertainties'])
    return ''

texts=[]
caption_numeric=[]
patterns = [r'forest|woodland|canopy',r'conifer|needle|pine|spruce',r'broad.leaf|deciduous',r'mixed forest',r'shrub|scrub|regenerat|clear.cut',r'heath|moor|sclerophyll|mediterranean',r'pasture|grassland|graz|meadow|fodder',r'orchard|vineyard|olive|permanent crop',r'field|agricultur|crop|arable',r'small.*field|fragment|mosaic',r'urban|settlement|residential|building',r'wetland|marsh|peat|bog|waterlog',r'river|lake|reservoir|canal|pond|stream',r'marine|coast|sea|ocean|estuar|lagoon',r'scattered trees|agroforest|agro.forestry',r'bare|soil|rock',r'snow|cloud|shadow']
for row in Crows:
    c=row['caption']
    text=string_values(c)
    # Landscape and summary convey more information than repeated grid vocabulary.
    text += ' '+string_values(c.get('global',{}))+' '+str(c.get('summary',''))
    texts.append(text)
    cov=c.get('global',{}).get('cover',[])
    cover=[0.]*7
    for entry in cov:
        name=entry.get('class','').lower()
        try: val=float(str(entry.get('percent',0)).replace('%',''))/100
        except (TypeError,ValueError):val=0
        for j,pat in enumerate([r'veget|forest|wood|tree',r'water',r'built|urban|settle|road',r'bare|soil|rock',r'forest|wood|tree',r'agric|crop|field',r'grass|shrub|scrub']):
            if re.search(pat,name):cover[j]+=val
    caption_numeric.append(cover + [np.log1p(len(re.findall(p,text.lower()))) for p in patterns])
C=np.array(caption_numeric)
vectorizer=TfidfVectorizer(min_df=3,max_df=.97,max_features=4500,ngram_range=(1,2),sublinear_tf=True,stop_words='english')
T=vectorizer.fit_transform(texts)

# All transformations use no labels; bank-fitted scaling preserves test distribution shifts.
Z=StandardScaler().fit(D[:n]).transform(D)
Z=np.clip(Z,-6,6)
# Resolve small water fractions, which are otherwise compressed close to zero.
Z=np.column_stack([Z, np.log1p(100*D[:,5]), np.sqrt(np.clip(D[:,5],0,1))])
Z=StandardScaler().fit(Z[:n]).transform(Z)
Cs=StandardScaler().fit(C[:n]).transform(C)
Cs=np.clip(Cs,-5,5)
Qs=Q-Q[:n].mean(0)
P=PCA(n_components=64,random_state=2026).fit(F[:n]).transform(F)

def lin(X):
    return X@X.T/X.shape[1]
def rbf(X,scale=1):
    dist=euclidean_distances(X,squared=True)
    med=np.median(dist[:n,:n][np.triu_indices(n,1)])
    return np.exp(-dist/(med*scale))

print('Loaded',n,'bank and',nt,'test patches; pseudo-label rates',Y.mean(0).round(3),flush=True)
KF=lin(F)
KF/=np.mean(np.diag(KF)[:n])
KT=(T@T.T).toarray()
KQ=lin(Qs)
KC=rbf(Cs)
KD=rbf(Z)
KR={s:rbf(F,s) for s in [.25,.5,1.,2.]}
kernels={
    'feature_linear':KF,
    **{'feature_rbf_'+str(s):k for s,k in KR.items()},
    'physical':KD,
    'text':KT,
    'answers':KQ,
    'evidence':.5*KQ+.35*KT+.15*KC,
    'image_physical':.75*KR[.5]+.25*KD,
    'combined':.55*KR[.5]+.2*KQ+.15*KT+.1*KD,
    'combined_linear':.5*KF+.25*KQ+.15*KT+.1*KD,
    'combined_text':.3*KR[.5]+.3*KQ+.3*KT+.1*KD,
}
folds=list(KFold(5,shuffle=True,random_state=20261119).split(np.arange(n)))
alphas=np.array([.03,.1,.3,1.,3.,10.])
predictions={}
test_predictions={}
details={}

def per_class_mse(y,p):
    return np.mean((y-p)**2,axis=0)

for name,kernel in kernels.items():
    # Constant kernel supplies a regularized intercept.
    kernel=kernel+1.
    oof=np.zeros((len(alphas),n,15))
    for train,val in folds:
        ev,U=eigh(kernel[np.ix_(train,train)],check_finite=False)
        uy=U.T@Y[train]
        ku=kernel[np.ix_(val,train)]@U
        for j,alpha in enumerate(alphas):
            oof[j,val]=np.clip(ku@(uy/(np.maximum(ev,0)+alpha)[:,None]),.005,.995)
    loss=np.mean((oof-Y[None,:,:])**2,axis=1)
    # Shrink class-specific selection toward global regularization, especially for rare classes.
    scaled=loss/(np.var(Y,axis=0)+.02)
    selection=.7*scaled+.3*scaled.mean(axis=1,keepdims=True)
    chosen=selection.argmin(axis=0)
    predictions[name]=np.column_stack([oof[chosen[c],:,c] for c in range(15)])
    ev,U=eigh(kernel[:n,:n],check_finite=False)
    uy=U.T@Y
    ku=kernel[n:,:n]@U
    test_predictions[name]=np.column_stack([np.clip(ku@(uy[:,c]/(np.maximum(ev,0)+alphas[chosen[c]])),.005,.995) for c in range(15)])
    details[name]={'mse':per_class_mse(Y,predictions[name]).tolist(),'alphas':alphas[chosen].tolist()}
    print(name,'MSE',np.mean((Y-predictions[name])**2).round(5),'alphas',alphas[chosen],flush=True)

# Direct class-wise answer calibration; beta-style shrinkage stabilizes rare answer categories.
oof=np.empty_like(Y)
for train,val in folds:
    for j in range(15):
        for cat in range(4):
            sel=train[Qcat[train,j]==cat]
            v=(Y[sel,j].sum()+6*Y[train,j].mean())/(len(sel)+6)
            oof[val[Qcat[val,j]==cat],j]=v
cal=np.empty((nt,15))
for j in range(15):
    for cat in range(4):
        sel=np.where(Qcat[:n,j]==cat)[0]
        cal[Qcat[n:,j]==cat,j]=(Y[sel,j].sum()+6*Y[:,j].mean())/(len(sel)+6)
predictions['calibration']=oof
test_predictions['calibration']=cal
print('calibration MSE',np.mean((oof-Y)**2).round(5),flush=True)

for name,X,leaf,mfeat in [
    ('trees_image',np.column_stack([P,D]),4,.8),
    ('trees_all',np.column_stack([P,D,Q,C]),4,.8),
    ('trees_evidence',np.column_stack([D,Q,C]),5,.9),
]:
    oof=np.empty_like(Y)
    for train,val in folds:
        model=ExtraTreesRegressor(n_estimators=350,min_samples_leaf=leaf,max_features=mfeat,n_jobs=2,random_state=104)
        model.fit(X[train],Y[train]);oof[val]=model.predict(X[val])
    model.fit(X[:n],Y)
    predictions[name]=oof
    test_predictions[name]=model.predict(X[n:])
    print(name,'MSE',np.mean((Y-oof)**2).round(5),flush=True)

np.savez_compressed(ROOT/'model_predictions.npz',Y=Y,hard=hard,**{'oof__'+k:v for k,v in predictions.items()},**{'test__'+k:v for k,v in test_predictions.items()})
(ROOT/'model_diagnostics.json').write_text(json.dumps(details,indent=2))

# Pool correlated kernel variants before stacking so that they do not count as independent votes.
groups={
    'image':['feature_linear','feature_rbf_0.25','feature_rbf_0.5','feature_rbf_1.0','feature_rbf_2.0','image_physical'],
    'joint':['combined','combined_linear','combined_text'],
    'text':['text','evidence','answers'],
    'trees':['trees_image','trees_all','trees_evidence'],
    'calibration':['calibration'],
    'physical':['physical'],
}
OP=[];TP=[]
for g,names in groups.items():
    losses=np.array([per_class_mse(Y,predictions[k]) for k in names])
    normalized=losses/(np.var(Y,axis=0)+.02)
    weights=np.exp(-(normalized-normalized.min(axis=0))/.06)
    weights/=weights.sum(axis=0)
    OP.append(np.sum(np.array([predictions[k] for k in names])*weights[:,None,:],axis=0))
    TP.append(np.sum(np.array([test_predictions[k] for k in names])*weights[:,None,:],axis=0))
OP=np.stack(OP,axis=-1); TP=np.stack(TP,axis=-1)
prior=np.array([.25,.35,.10,.20,.05,.05])
scores=np.empty((nt,15)); blended=np.empty_like(Y);weights=[]
for j in range(15):
    x=OP[:,j,:]; y=Y[:,j]
    strength=.006
    fun=lambda w: np.mean((x@w-y)**2)+strength*np.sum((w-prior)**2)
    result=minimize(fun,prior,method='SLSQP',bounds=[(0,1)]*len(prior),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'ftol':1e-12})
    assert result.success
    w=result.x
    weights.append(w.tolist())
    scores[:,j]=TP[:,j,:]@w
    blended[:,j]=x@w
    print(classes[j]['name'][:25], 'MSE',round(np.mean((blended[:,j]-y)**2),4),'AUC',round(roc_auc_score(hard[:,j],blended[:,j]),3),'weights',w.round(2),'test mean',round(scores[:,j].mean(),3),flush=True)

np.save(ROOT/'scores.npy',np.clip(scores,.005,.995).astype(np.float32))
np.savez_compressed(ROOT/'ensemble_diagnostics.npz',oof=blended,Y=Y,hard=hard,weights=weights)
assert np.load(ROOT/'scores.npy').shape==(997,15)
print('Wrote scores.npy; finite:',np.isfinite(scores).all(),'range',scores.min(),scores.max(),flush=True)
