"""Mapa de lesion por clasificacion de parches con solapamiento."""
import numpy as np, time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from extractores import textura

def rejilla(img, P, stride):
    """Coordenadas de la esquina superior izquierda de cada parche."""
    H,W = img.shape[:2]
    ys = list(range(0, H-P+1, stride)); xs = list(range(0, W-P+1, stride))
    return ys, xs

def entrenar(img, tumor, forma, P=64, n=400, semilla=3):
    """Entrena un clasificador de parches con etiquetas puras (>90% de una clase)."""
    rng=np.random.default_rng(semilla)
    Xs=[]; ys_=[]
    intentos=0
    while len(Xs)<n and intentos<n*80:
        intentos+=1
        y=rng.integers(0,img.shape[0]-P); x=rng.integers(0,img.shape[1]-P)
        ft=forma[y:y+P,x:x+P].mean(); tt=tumor[y:y+P,x:x+P].mean()
        if ft<0.95: continue
        if tt>0.90: Xs.append(img[y:y+P,x:x+P]); ys_.append(1)
        elif tt<0.02 and len([z for z in ys_ if z==0])<n//2:
            Xs.append(img[y:y+P,x:x+P]); ys_.append(0)
    X=np.array(Xs); yv=np.array(ys_)
    F=textura(X)
    clf=make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))
    clf.fit(F,yv)
    return clf, np.bincount(yv)

def construir(img, forma, clf, P=64, stride=32):
    """Mapa de probabilidad tumoral por acumulacion de parches solapados."""
    H,W=img.shape[:2]
    acc=np.zeros((H,W),np.float32); cnt=np.zeros((H,W),np.float32)
    ys,xs = rejilla(img,P,stride)
    lote=[]; pos=[]
    for y in ys:
        for x in xs:
            if forma[y:y+P,x:x+P].mean()<0.5: continue
            lote.append(img[y:y+P,x:x+P]); pos.append((y,x))
    if not lote: return acc, 0
    t0=time.time()
    prob=clf.predict_proba(textura(np.array(lote)))[:,1]
    dt=time.time()-t0
    for (y,x),p in zip(pos,prob):
        acc[y:y+P,x:x+P]+=p; cnt[y:y+P,x:x+P]+=1
    return np.divide(acc,np.maximum(cnt,1)), len(lote), dt

def dice(a,b):
    inter=(a&b).sum()
    return 2*inter/max(a.sum()+b.sum(),1)
