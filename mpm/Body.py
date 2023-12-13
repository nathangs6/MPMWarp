import numpy as np

class Body:
    _name: str
    _velocity: np.array
    _friction_coefficient: float
    _is_sticky: bool
    _defining_function: callable
    _normal_generator: callable
    _mesh: dict

    def __init__(self, name, mu, v: np.array, defining_function: callable, normal_generator: callable, mesh: dict, is_sticky=False) -> None:
        self._name = name
        self._velocity = v
        self._friction_coefficient = mu
        self._is_sticky = is_sticky
        self._defining_function = defining_function
        self._normal_generator = normal_generator
        self._mesh = mesh

    def get_name(self):
        return self._name

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

    def get_mesh(self):
        return self._mesh


class Plane(Body):
    def __init__(self, name, mu, v: np.array, n: np.array, x0: np.array, extent: float) -> None:
        """
        Returns the plane corresponding to the equation:
            q[0]x + q[1]y + q[2]z + d = 0
        """
        n /= np.linalg.norm(n)
        def plane_function(p):
            return np.dot(n, p-x0)

        def plane_normal(p):
            return n
        t1 = np.random.rand(3)
        t1 -= t1.dot(n) * n
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        mesh = {
            "vertices": np.array([x0,
                                  x0+extent*t1,
                                  x0-extent*t1,
                                  x0+extent*t2,
                                  x0-extent*t2]),
            "faces": np.array([[0,1,3], [0,1,4], [0,2,3], [0,2,4]])
        }
        Body.__init__(self, name, mu, v, plane_function, plane_normal, mesh)
