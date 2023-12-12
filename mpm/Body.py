import numpy as np

class Body:
    _velocity: np.array
    _friction_coefficient: float
    _is_sticky: bool
    _defining_function: callable
    _normal_generator: callable

    def __init__(self, mu, v: np.array, defining_function: callable, normal_generator: callable, is_sticky=False) -> None:
        self._velocity = v
        self._friction_coefficient = mu
        self._is_sticky = is_sticky
        self._defining_function = defining_function
        self._normal_generator = normal_generator

    def get_velocity(self):
        return np.copy(self._velocity)

    def set_velocity(self, v: np.array):
        self._velocity = v

    def get_mu(self):
        return self._friction_coefficient

    def is_sticky(self):
        return self._is_sticky

    def check_collision(self, p: np.array) -> bool:
        return self._defining_function(p) <= 0

    def get_normal(self, p: np.array) -> np.array:
        return self._normal_generator(p)


class Plane(Body):
    def __init__(self, mu, v: np.array, n: np.array, x0: np.array) -> None:
        """
        Returns the plane corresponding to the equation:
            q[0]x + q[1]y + q[2]z + d = 0
        """
        def plane_function(p):
            return np.dot(n, p-x0)

        def plane_normal(p):
            return n
        Body.__init__(self, mu, v, plane_function, plane_normal)
