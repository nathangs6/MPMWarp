import numpy as np
import numpy.linalg as npl
import warp as wp

@wp.func
def lame_parameter(c: float, zeta: float, JP: float):
    return c * wp.exp(zeta * (1.0 - JP))

@wp.func
def compute_stress(FE: wp.mat33, FP: wp.mat33, mu0: float, lam0: float, zeta: float) -> wp.mat33:
    JE      = wp.determinant(FE)
    JP      = wp.determinant(FP)
    mu      = lame_parameter(mu0, zeta, JP)
    lam     = lame_parameter(lam0, zeta, JP)
    U       = wp.mat33(dtype=wp.float32)
    S       = wp.vec3(dtype=wp.float32)
    V       = wp.mat33(dtype=wp.float32)
    wp.svd3(FE, U, S, V)
    RE      = U * wp.transpose(V)
    return (2.0*mu*(FE-RE)*wp.transpose(FE) + lam*(JE-1.0)*JE*wp.identity(n=3, dtype=wp.float32))/JP

@wp.kernel
def get_stresses(stress: wp.array(dtype=wp.mat33),
                 FE: wp.array(dtype=wp.mat33),
                 FP: wp.array(dtype=wp.mat33),
                 mu0: float,
                 lam0: float,
                 zeta: float) -> None:
    p = wp.tid()
    stress[p] = compute_stress(FE[p], FP[p], mu0, lam0, zeta)

@wp.func
def sum_stresses(volume: wp.array(dtype=wp.float32),
                 stress: wp.array(dtype=wp.mat33),
                 grad_wi: wp.array(dtype=wp.vec3)) -> wp.vec3:
    result = wp.vec3(0.0, 0.0, 0.0)
    for p in range(volume.shape[0]):
        result += volume[p] * stress[p] * grad_wi[p]
    return result

@wp.kernel
def compute_grid_forces(
    grid_forces:    wp.array(dtype=wp.vec3),
    volume:         wp.array(dtype=wp.float32),
    grad_wip:       wp.array(dtype=wp.vec3, ndim=2),
    stress:         wp.array(dtype=wp.mat33)
):
    """
    Compute the grid forces.
    """
    i = wp.tid()
    grid_forces[i] = -1.0 * sum_stresses(volume, stress, grad_wip[i])

@wp.kernel
def add_force(force: wp.array(dtype=wp.vec3),
              new_force: wp.vec3) -> None:
    i = wp.tid()
    force[i] = force[i] + new_force

@wp.kernel
def update_grid_velocities_with_ext_forces(new_v: wp.array(dtype=wp.vec3),
                                           old_v: wp.array(dtype=wp.vec3),
                                           mass: wp.array(dtype=wp.float32),
                                           ext_f: wp.array(dtype=wp.vec3),
                                           dt: float) -> None:
    i = wp.tid()
    new_v[i] = old_v[i] + ext_f[i] * dt / mass[i]



@wp.kernel
def solve_grid_velocity_explicit(new_v: wp.array(dtype=wp.vec3),
                                 old_v: wp.array(dtype=wp.vec3)) -> None:
    i = wp.tid()
    new_v[i] = old_v[i]
