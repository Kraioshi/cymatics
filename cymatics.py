import numpy as np
import math
import matplotlib.pyplot as plt


def frequency_to_modes(frequency: float, speed_of_sound: float, l: float, difference: int) -> tuple:
    """Calculates theoretical values for modes (m, n) for a given frequency on a grid l with the theoretical difference"""
    k = speed_of_sound / frequency
    n = int(np.round(k / (2 * np.pi / l)))
    m = n + difference
    return m, n


def create_chladni_pattern(grid_size: int, frequency: float, speed_of_sound: float, diff: int) -> np.ndarray:
    """Creates continuous chladni patterns based on the equation for the zeros of a standing wave on a square of the
    Chladni plate of side length L constrained at the center
    cos(n pi x/L) cos(m pi y / L) - cos(m pi x / L) cos(n pi y / L) = 0 """

    L = grid_size
    n, m = frequency_to_modes(frequency, speed_of_sound, math.sqrt(L), diff)

    x = np.linspace(0, L, grid_size)
    y = np.linspace(0, L, grid_size)
    x_mesh, y_mesh = np.meshgrid(x, y)

    # get the wiggle-wiggle, the booty displacement of the particle
    cymatics = (
            np.cos(n * (x_mesh * np.pi / L)) * np.cos(m * (y_mesh * np.pi / L)) -
            np.cos(m * (x_mesh * np.pi / L)) * np.cos(n * (y_mesh * np.pi / L))
    )

    # if cymatics abs value is < 0.1, wiggle-wigle is close to 0 (no shit, innit?)
    pattern = np.where(np.abs(cymatics) < 0.1, 0, 255)

    return pattern


def generate_grayscale(chladni_pattern, frequency):
    plt.imshow(chladni_pattern, cmap='gray', interpolation='nearest')
    plt.title(f'Chladni cymatic pattern for: {frequency} Hz')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    grid_size_q = 500
    base_frequency = 5201  # Frequency
    speed = 5210  # Speed of sound in iron

    pattern_chladni = create_chladni_pattern(grid_size_q, base_frequency, speed, diff=2)
    generate_grayscale(pattern_chladni, base_frequency)
