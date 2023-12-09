import warp as wp
import numpy as np

@wp.func
def N(x: float) -> float:
    """
    Computes the value of N(x).
    """
    if (0 <= wp.abs(x) and wp.abs(x) < 1):
        return (1.0/2.0) * wp.abs(x)**3.0 - x**2.0 + 2.0/3.0
    if (1 <= wp.abs(x) and wp.abs(x) < 2):
        return -(1.0/6.0) * wp.abs(x)**3.0 + x**2.0 - 2.0 * wp.abs(x) + 4.0/3.0
    return 0.0

@wp.func
def dN(x: float) -> float:
    if (0 <= wp.abs(x) and wp.abs(x) < 1):
        return (3.0/2.0)*wp.abs(x)*x - 2.0*x
    if (1 <= wp.abs(x) and wp.abs(x) < 2):
        return -wp.abs(x)*x/2.0 + 2.0*x - 2.0*x/wp.abs(x)
    return 0.0

@wp.func
def grid_basis_function(xp: wp.vec3, idx: wp.vec3, h: float) -> float:
    return N((xp[0]/h - idx[0])) * N((xp[1]/h - idx[1])) * N((xp[2]/h - idx[2]))

@wp.func
def grad_grid_basis_function(xp: wp.vec3, idx: wp.vec3, h: float) -> wp.vec3:
    wip_grad = wp.vec3(0.0,0.0,0.0)
    wip_grad[0] = N(xp[1]/h-idx[1]) * N(xp[2]/h - idx[2]) * dN(xp[0]/h - idx[0]) / h
    wip_grad[1] = N(xp[0]/h-idx[0]) * N(xp[2]/h - idx[2]) * dN(xp[1]/h - idx[1]) / h
    wip_grad[2] = N(xp[0]/h-idx[0]) * N(xp[1]/h - idx[1]) * dN(xp[2]/h - idx[2]) / h
    return wip_grad

@wp.kernel
def construct_interpolation(wip: wp.array(dtype=float, ndim=2), index: wp.array(dtype=wp.vec3), p: wp.array(dtype=wp.vec3), h: float) -> None:
    i, j = wp.tid()
    wip[i,j] = grid_basis_function(p[j], index[i], h)

@wp.kernel
def construct_interpolation_grad(wip_grad: wp.array(dtype=wp.vec3, ndim=2), index: wp.array(dtype=wp.vec3), p: wp.array(dtype=wp.vec3), h: float) -> None:
    i, j = wp.tid()
    wip_grad[i,j] = grad_grid_basis_function(p[j], index[i], h)


if __name__ == "__main__":
    wp.init()

    wip = wp.empty(shape=(3,4), dtype=float, device="cpu")
    np_indices = np.array([
        [0,0,0],
        [2,4,1],
        [4,-3,3]], dtype=np.float32)
    i = wp.from_numpy(np_indices, dtype=wp.vec3, device="cpu")
    np_points = np.array([
        [0,0,0],
        [1,3,-2],
        [0.1,0.2,0.3],
        [0.4,-0.2,0.35]], dtype=np.float32)
    p = wp.from_numpy(np_points, dtype=wp.vec3, device="cpu")
    h = 0.1
    wp.launch(kernel=construct_interpolation,
              dim=[3,4],
              inputs=[wip, i, p, h],
              device="cpu")
    print(wip)
