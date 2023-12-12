import os
import sys
curr_directory = os.path.dirname(__file__)
src_dir = os.path.join(curr_directory, "mpm")
sys.path.append(src_dir)
from MPM import MPM
from Scene import Scene
from Body import Plane
import numpy as np
import polyscope as ps
test_scene = Scene(theta_c=2.5e-2,
                   theta_s=7.5e-3,
                   hardening_coefficient=10.0,
                   mu0=1.0,
                   lam0=1.0,
                   initial_density=4e2,
                   initial_young_modulus=1.4e5,
                   poisson_ratio=0.2,
                   alpha=0.95,
                   spacing=0.1,
                   dt=0.1,
                   mass=np.array([1.0, 2.0, 3.0]),
                   position=np.array([
                       [0.0,0.0,0.0],
                       [0.1,-0.2,0.3],
                       [1.1,-2.2,0.01]]),
                   velocity=np.array([
                       [0.0,0.0,0.0],
                       [-0.3, 0.45,1.0],
                       [3.0,2.0,1.0]]),
                   grid=np.array([
                       [0.0,0.0,0.0],
                       [1.0,1.0,1.0],
                       [-1.0,1.0,-1.0]]),
                   bodies=np.array([Plane(0.1,
                                          np.array([0.0,0.0,0.0]),
                                          np.array([1.0,0.0,0.0]),
                                          np.array([0.0,0.0,0.0]))])
                   )
test_mpm = MPM(test_scene)
ps.init()
ps.register_point_cloud("my points", test_mpm.animate())

def callback():
    test_mpm.run(num_steps=10)
    ps.register_point_cloud("my points", test_mpm.animate())
    print("Executing callback")

ps.set_user_callback(callback)
ps.show()
ps.clear_user_callback()
