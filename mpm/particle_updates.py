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
def update_particle_velocity(new_vp: wp.array(dtype=wp.vec3),
                             old_vp: wp.array(dtype=wp.vec3),
                             new_vg: wp.array(dtype=wp.vec3),
                             old_vg: wp.array(dtype=wp.vec3),
                             wpi: wp.array(dtype=wp.float32, ndim=2),
                             a: float) -> None:
    """
    Update particle velocities using the grid velocities.
    """
    p = wp.tid()
    new_vp[p] = compute_vW(new_vg, wpi[p]) + a*(old_vp[p] - compute_vW(old_vg, wpi[p]))

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
