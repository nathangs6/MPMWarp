import numpy as np
import warp as wp

@wp.func
def sum_points_m(m: wp.array(dtype=wp.float32),
                 wi: wp.array(dtype=wp.float32)) -> wp.float32:
    result = float(0.0)
    for p in range(m.shape[0]):
        result += m[p] * wi[p]
    return result

@wp.kernel
def rasterize_mass(m_p: wp.array(dtype=wp.float32),
                   wip: wp.array(dtype=wp.float32, ndim=2),
                   m_g: wp.array(dtype=wp.float32)) -> None:
    i = wp.tid()
    m_g[i] = sum_points_m(m_p, wip[i])


@wp.func
def sum_points_v(m: wp.array(dtype=wp.float32),
                 v: wp.array(dtype=wp.vec3),
                 wi: wp.array(dtype=wp.float32)) -> wp.vec3:
    result = wp.vec3(0.0,0.0,0.0)
    for p in range(m.shape[0]):
        result += m[p] * wi[p] * v[p]
    return result

@wp.kernel
def rasterize_velocity(m_p: wp.array(dtype=wp.float32),
                       v_p: wp.array(dtype=wp.vec3),
                       wip: wp.array(dtype=wp.float32, ndim=2),
                       m_g: wp.array(dtype=wp.float32),
                       v_g: wp.array(dtype=wp.vec3)) -> None:
    i = wp.tid()
    v_g[i] = sum_points_v(m_p, v_p, wip[i]) / m_g[i]


if __name__ == "__main__":
    print("hello")
    wp.init()
    mass = wp.array([1,1,1], dtype=wp.float32)
    wip = wp.array([
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ], dtype=wp.float32)
    grid_mass = wp.empty_like(mass)
    wp.launch(
        kernel=rasterize_mass_warp,
        dim=3,
        inputs=[mass, wip, grid_mass],
        device="cpu"
    )
    print(grid_mass)
