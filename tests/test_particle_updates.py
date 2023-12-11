import os
import sys
test_directory = os.path.dirname(__file__)
src_dir = os.path.join(test_directory, '..', 'mpm')
sys.path.append(src_dir)
import particle_updates as src
import warp as wp
import numpy as np
TOL = 0.00001

def test_update_particle_position():
    """
    Test update_particle_position.
    """
    p = [wp.vec3(0.0,0.0,0.0),
         wp.vec3(1.0,2.0,-3.0),
         wp.vec3(0.5,-2.2,4.3)]
    v = [wp.vec3(1.0,-1.0,0.0),
         wp.vec3(2.2,3.5,-6.0),
         wp.vec3(0.0,0.0,0.0)]
    p = wp.array(p, dtype=wp.vec3, device="cpu")
    v = wp.array(v, dtype=wp.vec3, device="cpu")
    dt = 0.1
    wp.launch(kernel=src.update_particle_position,
              dim=3,
              inputs=[p, v, dt],
              device="cpu")
    actual = np.array(p)
    expected = [np.array([0.1,-0.1,0.0]),
                np.array([1.22,2.35,-3.6]),
                np.array([0.5,-2.2,4.3])]
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL

def test_update_particle_velocity():
    """
    Test update_particle_velocity.
    """
    old_vp = np.array([wp.vec3(1.0,0.0,2.0)])
    new_vg = np.array([wp.vec3(1.0,2.0,3.0),
                       wp.vec3(0.3,0.2,0.1),
                       wp.vec3(2.2,-3.5,1.3)])
    old_vg = np.array([wp.vec3(0.0,0.0,0.0),
                       wp.vec3(3.0,2.0,1.0),
                       wp.vec3(0.1,0.2,0.3)])
    wip = np.array([
        [1.0],
        [0.3],
        [2.2]])
    wpi = wip.transpose()
    a = 0.1

    old_vp = wp.array(old_vp, dtype=wp.vec3, device="cpu")
    new_vg = wp.array(new_vg, dtype=wp.vec3, device="cpu")
    old_vg = wp.array(old_vg, dtype=wp.vec3, device="cpu")
    wpi = wp.array(wpi, dtype=wp.float32, device="cpu")
    new_vp = wp.empty_like(old_vp)
    wp.launch(kernel=src.update_particle_velocity,
              dim=2,
              inputs=[new_vp, old_vp, new_vg, old_vg, wpi, a],
              device="cpu")
    actual = np.array(new_vp)
    expected = [np.array([5.918,-5.744,5.994])]
    for i in range(len(expected)):
        assert np.linalg.norm(actual[i] - expected[i]) <= TOL
