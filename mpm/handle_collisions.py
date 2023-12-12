import numpy as np
import numpy.linalg as npl

def compute_vrel(v: np.array, vco: np.array) -> np.array:
    return v - vco

def compute_vn(vrel: np.array, n: np.array) -> float:
    return np.dot(vrel, n)

def compute_vt(vrel: np.aray, n: np.array, vn: float):
    return vrel - vn * n

def compute_collision(p: np.array, v: np.array, body: Body) -> np.array:
    vco = body.get_velocity()
    n = body.get_normal(p)
    mu = body.get_friction()
    vrel = compute_vrel(v, vco)
    vn = compute_vn(vrel, n)
    if vn >= 0:
        return v
    vt = compute_vt(vrel, n, vn)
    if npl.norm(vt) <= -mu*vn:
        vrel = np.array([0.0,0.0,0.0])
    else:
        vrel = vt + mu*vn*vt/npl.norm(vt)
    return vrel + vco
