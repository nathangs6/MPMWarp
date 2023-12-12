import numpy as np

class Body:
    _velocity: np.array
    _friction_coefficient: float
    _defining_function: callable
    _normal_generator: callable

    def __init__(self, v: np.array, mu, defining_function: callable, normal_generator: callable) -> None:
        self._velocity = v
        self._friction_coefficient: mu
        self._defining_function = defining_function
        self._normal_generator = normal_generator

    def get_velocity():
        return np.copy(self._velocity)

    def set_velocity(v: np.array):
        self._velocity = v

    def get_mu():
        return self._friction_coefficient

    def check_collision(p: np.array) -> bool:
        return self._defining_function(p) <= 0

    def get_normal(p: np.array) -> np.array:
        return self._normal_generator(p)


class Plane(Body):
    def __init__(self, v: np.array, n: np.array, x0: np.array) -> None:
        """
        Returns the plane corresponding to the equation:
            q[0]x + q[1]y + q[2]z + d = 0
        """
        def plane_function(p):
            return np.dot(n, p-x0)

        def plane_normal(p):
            return n
        Body.__init__(v, plane_function, plane_normal)
