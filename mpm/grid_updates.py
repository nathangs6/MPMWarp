import numpy as np
import numpy.linalg as npl
import warp as wp

def lame_parameter(c: float, zeta: float, JP: float):
    return c * np.exp(zeta * (1 - JP))

def elasto_plastic_energy_density(
    FE:     np.array,
    FP:     np.array,
    mu0:    float,
    lam0:   float,
    zeta:   float
):
    JE      =   npl.det(FE)
    JP      =   npl.det(FP)
    mu      =   lame_parameter(mu0, zeta, JP)
    lam     =   lame_parameter(lam0, zeta, JP)
    RE, SE  =   npl.polar(FE)

    return mu * npl.norm(FE - RE)**2 + (lam/2) * (JE - 1)**2

def d_elasto_plastic_energy_density(FE: np.array, mu0: float, zeta: float):
    RE, SE = npl.polar(FE)
    return 2*mu*(FE - RE)


def compute_grid_forces(
    grid_forces:    np.array,
    volume:         np.array,
    grad_wip:       np.array,
    FE:             np.array,
    FP:             np.array,
    mu0:            float,
    lam0:           float,
    zeta:           float
):
    """
    Compute the grid forces.
    """
    for i in range(len(grid_forces)):
        grid_forces[i] = 0.0
        for p in range(len(FE)):
            stress = (1/npl.det(FP[p])) * d_elasto_plastic_energy_density(FE[p], mu0, zeta) * np.transpose(FE[p])
            grid_forces[i] -= volume[p] * stress * grad_wip[i][p]


@wp.kernel
def update_grid_velocities_with_ext_forces(new_v: wp.array(dtype=wp.vec3),
                                           old_v: wp.array(dtype=wp.vec3),
                                           mass: wp.array(dtype=wp.float32),
                                           ext_f: wp.array(dtype=wp.vec3),
                                           dt: float) -> None:
    i = wp.tid()
    new_v[i] = old_v[i] + ext_f[i] * dt / mass[i]



@wp.kernel
def solve_grid_velocity_explicit(new_v: wp.array(wp.vec3),
                                 old_v: wp.array(dtype=wp.float32)) -> None:
    i = wp.tid()
    new_v[i] = old_v[i]
