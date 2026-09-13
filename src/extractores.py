"""Tres familias de extractores de caracteristicas, de simple a sofisticado."""
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray, rgb2hsv

def color(X):
    """Estadisticas de color: lo mas simple que puede funcionar."""
    F=[]
    for im in X:
        a=im.astype(np.float32)/255
        h=rgb2hsv(a)
        F.append(np.r_[a.mean((0,1)), a.std((0,1)),
                       h.mean((0,1)), h.std((0,1)),
                       np.percentile(a.reshape(-1,3),[10,50,90],axis=0).ravel()])
    return np.array(F)

def textura(X, P=8, R=1.0):
    """Descriptores clasicos de textura: LBP + matriz de coocurrencia (Haralick)."""
    F=[]
    for im in X:
        g=(rgb2gray(im)*255).astype(np.uint8)
        lbp=local_binary_pattern(g, P, R, method='uniform')
        hist,_=np.histogram(lbp, bins=P+2, range=(0,P+2), density=True)
        glcm=graycomatrix(g//8, distances=[1,3], angles=[0,np.pi/4,np.pi/2,3*np.pi/4],
                          levels=32, symmetric=True, normed=True)
        props=[graycoprops(glcm,p).ravel() for p in
               ('contrast','homogeneity','energy','correlation')]
        F.append(np.r_[hist, np.concatenate(props)])
    return np.array(F)

_BANCO=None
def _banco(semilla=0, n=64, k=7):
    global _BANCO
    if _BANCO is None:
        rng=np.random.default_rng(semilla)
        f=rng.standard_normal((n,3,k,k)).astype(np.float32)
        f-=f.mean((1,2,3),keepdims=True)
        f/=(np.sqrt((f**2).sum((1,2,3),keepdims=True))+1e-8)
        _BANCO=f
    return _BANCO

def random_cnn(X, n=64, k=7):
    """Convoluciones ALEATORIAS + pooling estadistico.

    No hay entrenamiento ni preentrenamiento: mide cuanto aporta la
    arquitectura por si sola, antes de aprender nada.
    """
    from scipy.signal import fftconvolve
    f=_banco(n=n,k=k); F=[]
    for im in X:
        a=(im.astype(np.float32)/255)
        a=(a-a.mean((0,1)))/(a.std((0,1))+1e-6)
        v=[]
        for filtro in f:
            r=sum(fftconvolve(a[...,c], filtro[c][::-1,::-1], mode='valid')
                  for c in range(3))
            r=np.maximum(r,0)                      # ReLU
            v += [r.mean(), r.std(), r.max()]      # pooling estadistico
        F.append(np.array(v))
    return np.array(F)
