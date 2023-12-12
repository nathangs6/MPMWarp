import numpy as np
import warp as wp

@wp.kernel
def update_particle_position(
    position: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    dt: float
):
    """
    Update position using the old position and velocity.
    """
    p = wp.tid()
    position[p] = position[p] + dt * velocity[p]

@wp.func
def compute_vW(vg: wp.array(dtype=wp.vec3),
               wp: wp.array(dtype=wp.float32)) -> wp.vec3:
    result = wp.vec3(0.0, 0.0, 0.0)
    for i in range(vg.shape[0]):
        result += vg[i] * wp[i]
    return result

@wp.kernel
def update_particle_velocity(vp: wp.array(dtype=wp.vec3),
                             new_vg: wp.array(dtype=wp.vec3),
                             old_vg: wp.array(dtype=wp.vec3),
                             wpi: wp.array(dtype=wp.float32, ndim=2),
                             a: float) -> None:
    """
    Update particle velocities using the grid velocities.
    """
    p = wp.tid()
    vp[p] = compute_vW(new_vg, wpi[p]) + a*(vp[p] - compute_vW(old_vg, wpi[p]))

@wp.func
def compute_outer_vi_gradwp(vi: wp.array(dtype=wp.vec3),
                         grad_wp: wp.array(dtype=wp.vec3)) -> wp.mat33:
    result = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    for i in range(vi.shape[0]):
        result += wp.outer(vi[i], grad_wp[i])
    return result

@wp.kernel
def update_particle_F(f: wp.array(dtype=wp.mat33),
                      new_vi: wp.array(dtype=wp.vec3),
                      grad_wpi: wp.array(dtype=wp.vec3, ndim=2),
                      dt: float) -> None:
    p = wp.tid()
    f[p] = f[p] + dt * wp.mul(compute_outer_vi_gradwp(new_vi, grad_wpi[p]), f[p])


@wp.func
def clamp_vec3(A: wp.vec3, lower: float, upper: float) -> wp.vec3:
    result = wp.vec3(0.0,0.0,0.0)
    for i in range(3):
        result[i] = wp.clamp(A[i], lower, upper)
    return result

@wp.kernel
def update_particle_FE_FP(fe: wp.array(dtype=wp.mat33),
                          fp: wp.array(dtype=wp.mat33),
                          f: wp.array(dtype=wp.mat33),
                          new_vi: wp.array(dtype=wp.vec3),
                          grad_wpi: wp.array(dtype=wp.vec3, ndim=2),
                          dt: float,
                          theta_c: float,
                          theta_s: float) -> None:
    p = wp.tid()
    fe_p = fe[p]
    f_p = f[p]
    grad_wpi_p = grad_wpi[p]
    outer = compute_outer_vi_gradwp(new_vi, grad_wpi_p)
    fe_hat = fe_p + dt * wp.mul(outer, fe_p)
    U = wp.mat33()
    S = wp.vec3()
    V = wp.mat33()
    wp.svd3(fe_hat, U, S, V)
    S = clamp_vec3(S, 1.0-theta_c, 1.0+theta_s)
    fe[p] = U * wp.diag(S) * wp.transpose(V)
    fp[p] = V * wp.inverse(wp.diag(S)) * wp.transpose(U) * f_p


if __name__ == "__main__":
    wp.init()
    dt = 0.1
    position = wp.array([
        wp.vec3(0.0,0.0,0.0),
        wp.vec3(1.0,2.0,3.0)], dtype=wp.vec3)
    old_vg=wp.array([
        wp.vec3(0.1,0.2,0.3),
        wp.vec3(1.0,2.0,3.0),
        wp.vec3(1.0,1.0,1.0)], dtype=wp.vec3)
    new_vg=wp.array([
        wp.vec3(0.0,0.0,0.0),
        wp.vec3(1.0,1.0,1.0),
        wp.vec3(1.0,2.0,3.0)], dtype=wp.vec3)
    old_vp=wp.array([
        wp.vec3(1.0,1.0,1.0),
        wp.vec3(0.0,0.0,0.0)], dtype=wp.vec3)
    new_vp=wp.array([
        wp.vec3(0.0,0.0,0.0),
        wp.vec3(0.0,0.0,0.0)], dtype=wp.vec3)
    wpi = wp.array(np.transpose(np.array([
        [1.0,2.0],
        [3.0,4.0],
        [0.1,0.2]])), dtype=wp.float32)
    a = 0.5

    wp.launch(kernel=update_particle_velocity,
              dim=2,
              inputs=[new_vp, old_vp, new_vg, old_vg, wpi, a],
              device="cpu")
    print(new_vp)
    wp.launch(kernel=update_particle_position,
              dim=2,
              inputs=[position, new_vp, dt],
              device="cpu")
    print(position)

    fe = wp.array([wp.mat33(1.0,1.0,1.0,2.0,2.0,3.0,0.5,1.1,2.2),
                   wp.mat33(1.0,1.0,1.0,2.0,2.0,3.0,0.5,1.1,2.2)], dtype=wp.mat33)
    fp = wp.array([wp.mat33(-1.0,2.0,3.0,0.5,-1.1,3.6,2.3,7.1,0.1),
                   wp.mat33(-1.0,2.0,3.0,0.5,-1.1,3.6,2.3,7.1,0.1)], dtype=wp.mat33)
    f = wp.array([wp.mat33(1.0,-2.0,0.5, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0),
                  wp.mat33(1.0,-2.0,0.5, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0)], dtype=wp.mat33)
    new_vi = wp.array([wp.vec3(0.5,-1.1,3.4)], dtype=wp.vec3)
    grad_wpi = wp.array([[wp.vec3(1.0,2.0,3.0)], [wp.vec3(1.0,2.0,3.0)]], dtype=wp.vec3, ndim=2)
    dt = 0.1
    theta_c = 0.1
    theta_s = 0.2
    fe_hat = wp.zeros_like(fe)
    U = wp.zeros_like(fe)
    S = wp.zeros(shape=fe.shape, dtype=wp.vec3)
    V = wp.zeros_like(fe)
    wp.launch(kernel=update_particle_FE_FP,
              dim=2,
              inputs=[fe, fp, f, new_vi, grad_wpi, dt, theta_c, theta_s],
              device="cpu")
    print(fe)
