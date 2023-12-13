# Import Packages
import os
import sys
curr_directory = os.path.dirname(__file__)
src_dir = os.path.join(curr_directory, "mpm")
sys.path.append(src_dir)
from MPM import MPM
from Scene import BallDrop
import numpy as np
import polyscope as ps

# Set Parameters
save = False
num_frames = 100
radius = 0.1
def filename(frame: int, max_digits: int):
    image_num = "00000" + str(frame)
    return "output/image-"+image_num[-max_digits:]+".png"
max_digits = len(str(num_frames))

# Define MPM and Scene
scene = BallDrop()
mpm = MPM(scene)
def updatePS(mpm):
    ps.register_point_cloud("points", mpm.get_position(), radius=radius)
    for body in mpm.bodies:
        mesh = body.get_mesh()
        ps.register_surface_mesh(body.get_name(), mesh["vertices"], mesh["faces"])

# Setup Polyscope
ps.init()

## Define call back function: will run on every loop
def callback():
    if mpm.frame >= num_frames:
        ps.clear_user_callback()
        return
    mpm.step()
    updatePS(mpm)
    if save:
        ps.screenshot(filename=filename(mpm.frame, max_digits))
    print(str(mpm.frame) + ": " + str(min(mpm.get_position(), key=lambda point: point[2])))

## Setup visualization scene
ps.set_up_dir("z_up")
ps.set_automatically_compute_scene_extents(False)
ps.set_length_scale(1.0)
low = np.array((-1.0, -1.0, -0.01))
high = np.array((1.0, 1.0, 1.0))
ps.set_bounding_box(low, high)
ps.look_at((0.0, 4.0, 1.0), (0.0, 0.0, 0.5))

## Add user callback
ps.set_user_callback(callback)

# Run
mpm.init_animation()
updatePS(mpm)
ps.show()

# Clean up
ps.clear_user_callback()
