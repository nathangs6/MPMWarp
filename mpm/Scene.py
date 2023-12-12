import numpy as np

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
