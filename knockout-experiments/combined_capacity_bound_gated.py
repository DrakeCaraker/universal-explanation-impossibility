import warnings, numpy as np, json
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes, load_iris, fetch_california_housing
def ens(Mdl,X,y,M,seed):
    r=np.random.default_rng(seed); n=X.shape[0]
    return np.array([Mdl(random_state=int(r.integers(1e9))).fit(X[i],y[i]).feature_importances_ for i in (r.integers(0,n,n) for _ in range(M))])
def corr_hi(X):
    lam=np.sort(np.clip(np.linalg.eigvalsh(np.corrcoef(X,rowvar=False)),0,None))[::-1]; return int((lam>=0.5).sum())
def extra_gated(X,co):
    p=X.shape[1]; C=np.abs(np.corrcoef(X,rowvar=False)); mi=np.abs(co).mean(0); floor=np.median(mi)
    parent=list(range(p))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    nodes=set()
    for j in range(p):
        for k in range(j+1,p):
            D=co[:,j]-co[:,k]; snr=abs(D.mean())/(D.std()+1e-12)
            if C[j,k]<0.3 and snr<0.5 and mi[j]>=floor and mi[k]>=floor:   # IMPORTANCE GATE
                nodes.add(j); nodes.add(k); parent[find(j)]=find(k)
    if not nodes: return 0
    return len(nodes)-len({find(x) for x in nodes})
out={}
r=np.random.default_rng(0); n=800; p=8
Xn=r.standard_normal((n,p)); y=(Xn[:,0]+Xn[:,1])+0.6*Xn[:,2]+0.3*Xn[:,3]+0.3*r.standard_normal(n); Xn=(Xn-Xn.mean(0))/Xn.std(0)
co=ens(lambda **k:RandomForestRegressor(n_estimators=60,max_depth=6,**k),Xn,y,60,2)
out["stress"]={"feat_hi":corr_hi(Xn),"extra_gated":extra_gated(Xn,co),"combined":corr_hi(Xn)-extra_gated(Xn,co)}
real=[]
for name,loader,clf in [("breast_cancer",load_breast_cancer,True),("wine",load_wine,True),("diabetes",load_diabetes,False),("iris",load_iris,True),("california",fetch_california_housing,False)]:
    d=loader(); X=np.asarray(d.data,float); y=np.asarray(d.target)
    if name=="california": X,y=X[:1000],y[:1000]
    X=(X-X.mean(0))/(X.std(0)+1e-12); p=X.shape[1]
    Mdl=(lambda **k:RandomForestClassifier(n_estimators=50,max_depth=6,**k)) if clf else (lambda **k:RandomForestRegressor(n_estimators=50,max_depth=6,**k))
    co=ens(Mdl,X,y,40,3); hi=corr_hi(X); ex=extra_gated(X,co)
    real.append({"ds":name,"p":p,"feat_hi":hi,"extra_gated":ex,"combined":hi-ex})
out["real"]=real
print(json.dumps(out,indent=2))
