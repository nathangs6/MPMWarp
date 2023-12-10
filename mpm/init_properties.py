import numpy as np

def init_cell_density(cell_density: np.array,
                      m_grid: np.array,
                      h: float) -> None:
    np.copyto(cell_density, m_grid / h**3)

def init_particle_density(particle_density: np.array,
                          m_grid: np.array,
                          w: np.array,
                          h: float) -> None:
    np.copyto(particle_density, np.matmul(m_grid.transpose(), w).transpose()/h**3)


def init_particle_volume(
    particle_volume: np.array,
    particle_mass: np.array,
    particle_density: np.array
):
    np.copyto(particle_volume, particle_mass / particle_density)
