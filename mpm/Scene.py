import numpy as np
from Body import Plane

class Scene:
    theta_c: float
    theta_s: float
    hardening_coefficient: float
    mu0: float
    lam0: float
    initial_density: float
    initial_young_modulus: float
    poission_ratio: float
    alpha: float
    spacing: float
    dt: float
    mass: np.array
    position: np.array
    velocity: np.array
    grid: np.array
    bodies: np.array

    def __init__(self, theta_c, theta_s, hardening_coefficient, mu0, lam0, initial_density, initial_young_modulus, poisson_ratio, alpha, spacing, dt, mass, position, velocity, grid, bodies):
        self.theta_c = theta_c
        self.theta_s = theta_s
        self.hardening_coefficient = hardening_coefficient
        self.mu0 = mu0
        self.lam0 = lam0
        self.initial_density = initial_density
        self.initial_young_modulus = initial_young_modulus
        self.poisson_ratio = poisson_ratio
        self.alpha = alpha
        self.spacing = spacing
        self.dt = dt
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.grid = grid
        self.bodies = bodies

class TestScene(Scene):
    def __init__(self):
        grid = []
        for i in [-1.0,0.0,1.0]:
            for j in [-1.0,0.0,1.0]:
                for k in [0.0,2.0]:
                    grid.append([i,j,k])
        grid = np.array(grid)
        Scene.__init__(self,
                       theta_c=0.1,
                       theta_s=0.2,
                       hardening_coefficient=1.0,
                       mu0=1.0,
                       lam0=1.0,
                       initial_density=100.0,
                       initial_young_modulus=1e5,
                       poisson_ratio=0.2,
                       alpha=0.5,
                       spacing=1.0,
                       dt=1e-2,
                       mass=np.array([0.1, 0.1]),
                       position=np.array([[0.0,0.0,1.0], [0.0,0.0,2.0]]),
                       velocity=np.array([[0.0,0.0,0.0], [0.0,0.0,0.0]]),
                       grid=grid,
                       bodies = np.array([])
                       )


class BallDrop(Scene):
    def __init__(self):
        position = []
        mass = []
        num_points = 100
        r_vals = np.random.uniform(0, 1, num_points)
        t_vals = np.random.uniform(0.01, 2*np.pi-0.01, num_points)
        p_vals = np.random.uniform(0.01, np.pi-0.01, num_points)
        for i in range(len(r_vals)):
            position.append([r_vals[i]*np.sin(t_vals[i])*np.cos(p_vals[i]),
                             r_vals[i]*np.sin(t_vals[i])*np.sin(p_vals[i]),
                             r_vals[i]*np.cos(t_vals[i])+2.0
                             ])
            mass.append(0.1)
        position = np.array(position)
        mass = np.array(mass)
        velocity = np.zeros_like(position)
        grid = []
        for i in range(-10, 11):
            for j in range(-10, 11):
                for k in range(0, 11):
                    grid.append([i, j, k])
        grid = np.array(grid)
        bodies = np.array([Plane("floor",
                                 0.5,
                                 np.array([0.0,0.0,0.0]),
                                 np.array([0.0,0.0,1.0]),
                                 np.array([0.0,0.0,0.0]),
                                 5.0)])
        Scene.__init__(self,
                       theta_c=2.5e-2,
                       theta_s=7.5e-3,
                       hardening_coefficient=10.0,
                       mu0=1.0,
                       lam0=1.0,
                       initial_density=4e2,
                       initial_young_modulus=1.4e5,
                       poisson_ratio=0.2,
                       alpha=0.95,
                       spacing=0.1,
                       dt=1e-3,
                       mass=mass,
                       position=position,
                       velocity=velocity,
                       grid=grid,
                       bodies=bodies)
