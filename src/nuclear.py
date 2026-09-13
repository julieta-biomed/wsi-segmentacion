"""Segmentacion pixel a pixel por densidad nuclear, sin aprendizaje profundo."""
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def nucleos(img):
    """Detecta nucleos por deconvolucion de tincion H&E + watershed.

    rgb2hed separa los canales de hematoxilina (nucleos) y eosina (citoplasma),
    que es lo correcto en histologia: usar el gris mezcla ambas tinciones.
    """
    hed = rgb2hed(img.astype(np.float32)/255)
    h = hed[...,0]                                  # canal de hematoxilina
    h = (h-h.min())/(h.max()-h.min()+1e-9)
    mascara = h > threshold_otsu(h)
    mascara = ndi.binary_opening(mascara, np.ones((3,3)))
    # separar nucleos que se tocan
    dist = ndi.distance_transform_edt(mascara)
    picos = peak_local_max(dist, min_distance=4, labels=mascara)
    marcadores = np.zeros(dist.shape, int)
    marcadores[tuple(picos.T)] = np.arange(1, len(picos)+1)
    etiquetas = watershed(-dist, marcadores, mask=mascara)
    return etiquetas, mascara

def densidad(etiquetas, sigma=28):
    """Mapa continuo de densidad nuclear: nucleos por unidad de area."""
    centros = np.zeros(etiquetas.shape, np.float32)
    objs = ndi.find_objects(etiquetas)
    for k,sl in enumerate(objs):
        if sl is None: continue
        cy=(sl[0].start+sl[0].stop)//2; cx=(sl[1].start+sl[1].stop)//2
        centros[cy,cx]=1
    return gaussian(centros, sigma=sigma)
