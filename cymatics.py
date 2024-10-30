import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def calculate_modes(frequency: float, sound_speed: float, grid_size: int, mode_difference: int = 0) -> tuple[int, int]:
    """
    Calculates mode numbers (m, n) for a given frequency and grid size.

    :param frequency: The frequency of the vibration.
    :param sound_speed: Speed of sound in the material.
    :param grid_size: Size of the grid to determine the length of one side of the Chladni plate.
    :param mode_difference: Difference to offset the m mode from n.
    :return: Tuple of mode numbers (m, n).
    """
    length = np.sqrt(grid_size)  # Calculate side length based on grid size
    wavelength = sound_speed / frequency
    n_mode = int(np.round(wavelength / (2 * np.pi / length)))
    m_mode = n_mode + mode_difference
    return m_mode, n_mode


def compute_wave_pattern(grid_size: int, m: int, n: int) -> NDArray:
    """
    Generates a Chladni wave pattern based on the mode numbers and grid size
    based on the equation for the zeros of a standing wave on a square of the
    Chladni plate of side length L constrained at the center
    cos(n pi x/L) cos(m pi y / L) - cos(m pi x / L) cos(n pi y / L) = 0

    :param grid_size: Size of the square grid.
    :param m: Mode number along the x-axis.
    :param n: Mode number along the y-axis.
    :return: A 2D numpy array representing the wave pattern.
    """
    L = grid_size
    x = np.linspace(0, L, grid_size)
    y = np.linspace(0, L, grid_size)
    x_mesh, y_mesh = np.meshgrid(x, y)

    # Calculate the wave pattern
    wave_pattern = (
            np.cos(n * (x_mesh * np.pi / L)) * np.cos(m * (y_mesh * np.pi / L)) -
            np.cos(m * (x_mesh * np.pi / L)) * np.cos(n * (y_mesh * np.pi / L))
    )
    return wave_pattern


def generate_chladni_pattern(grid_size: int, frequency: float, sound_speed: float,
                             mode_difference: int = 0, threshold: float = 0.1) -> NDArray:
    """
    Creates a Chladni pattern by computing standing wave zeros on a constrained plate.

    :param grid_size: Size of the square grid.
    :param frequency: Vibration frequency.
    :param sound_speed: Speed of sound in the material.
    :param mode_difference: Difference between the m and n modes.
    :param threshold: Value below which wave patterns are set to zero for visualization.
    :return: A 2D numpy array with Chladni pattern values.
    """
    # Calculate theoretical modes based on input parameters
    m, n = calculate_modes(frequency, sound_speed, grid_size, mode_difference)

    # Generate wave pattern and apply threshold to create Chladni pattern
    wave_pattern = compute_wave_pattern(grid_size, m, n)
    pattern_chladni = np.where(np.abs(wave_pattern) < threshold, 0, 255)

    return pattern_chladni


def display_pattern(pattern: NDArray, frequency: float, title: str = None):
    """
    Displays the Chladni pattern as a grayscale image.

    :param pattern: A 2D numpy array representing the Chladni pattern.
    :param frequency: The frequency of the generated pattern.
    :param title: Optional custom title for the plot.
    """
    plt.imshow(pattern, cmap='gray', interpolation='nearest')
    plt.title(title if title else f'Chladni Cymatic Pattern at {frequency} Hz')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    MODE_DIFF = 2
    GRID_SIZE = 500
    # Frequency in Hz
    BASE_FREQUENCY = 5201
    # Iron = 5210 m/s, Air = 343 m/s
    SOUND_SPEED = 5210
    # Threshold for defining wave nodes
    THRESHOLD = 0.1

    # Generate and display Chladni pattern
    chladni_pattern = generate_chladni_pattern(GRID_SIZE, BASE_FREQUENCY, SOUND_SPEED, MODE_DIFF, THRESHOLD)
    display_pattern(chladni_pattern, BASE_FREQUENCY)
