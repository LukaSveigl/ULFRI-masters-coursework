import math
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from typing import List, Tuple

from ex4_utils import kalman_step

def random_walk(q: float, r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the state transition matrix A, the observation matrix C, 
    the system covariance Q, and the observation covariance R for a random walk model.

    The random walk model is defined by modeling the velocity as white noise.

    Args:
        q (float): The variance of the process noise.
        r (float): The variance of the observation noise.

    Returns:
        tuple: A tuple containing the state transition matrix A, the observation matrix C, 
        the system covariance Q, and the observation covariance R.
    """
    # The F and L matrices are derived from the following equation:
    #   x_{k+1} = F * x_k + L * w_k
    F = sp.Matrix([[0, 0], [0, 0]])
    L = sp.Matrix([[1, 0], [0, 1]])
    T = sp.symbols('T')

    # Compute the system covariance Q using the equation: Q = int(F * L * Q * L.T * F.T * dt).
    Fi = sp.exp(F * T)
    Q = np.array(sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, 1))).astype(np.float64)

    # Define the observation matrix H and the observation covariance R.
    R = np.array(sp.Matrix([[r, 0], [0, r]])).astype(np.float64)
    H = np.array(sp.Matrix([[1, 0], [0, 1]])).astype(np.float64)

    F = np.array(sp.exp(F)).astype(np.float64)

    return F, H, Q, R


def nearly_constant_velocity(q: float, r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the state transition matrix A, the observation matrix C,
    the system covariance Q, and the observation covariance R for a nearly constant velocity model.

    The nearly constant velocity model is defined by modeling the acceleration as white noise.

    Args:
        q (float): The variance of the process noise.
        r (float): The variance of the observation noise.

    Returns:
        tuple: A tuple containing the state transition matrix A, the observation matrix C,
        the system covariance Q, and the observation covariance R.
    """
    # The F and L matrices are derived from the following equation:
    #   x_{k+1} = F * x_k + L * w_k
    F = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    L = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])
    T = sp.symbols('T')

    # Compute the system covariance Q using the equation: Q = int(F * L * Q * L.T * F.T * dt).
    Fi = sp.exp(F * T)
    Q = np.array(sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, 1))).astype(np.float64)

    # Define the observation matrix H and the observation covariance R.
    R = np.array(sp.Matrix([[r, 0], [0, r]])).astype(np.float64)
    H = np.array(sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0]])).astype(np.float64)

    F = np.array(sp.exp(F)).astype(np.float64)

    return F, H, Q, R


def nearly_constant_acceleration(q: float, r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the state transition matrix A, the observation matrix C,
    the system covariance Q, and the observation covariance R for a nearly constant acceleration model.

    The nearly constant acceleration model is defined by modeling the jerk as white noise.

    Args:
        q (float): The variance of the process noise.
        r (float): The variance of the observation noise.

    Returns:
        tuple: A tuple containing the state transition matrix A, the observation matrix C,
        the system covariance Q, and the observation covariance R.
    """
    # The F and L matrices are derived from the following equation:
    #   x_{k+1} = F * x_k + L * w_k
    F = sp.Matrix([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    L = sp.Matrix([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]])
    T = sp.symbols('T')

    # Compute the system covariance Q using the equation: Q = int(F * L * Q * L.T * F.T * dt).
    Fi = sp.exp(F * T)
    Q = np.array(sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, 1))).astype(np.float64)

    # Define the observation matrix H and the observation covariance R.
    R = np.array(sp.Matrix([[r, 0], [0, r]])).astype(np.float64)
    H = np.array(sp.Matrix([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])).astype(np.float64)

    F = np.array(sp.exp(F)).astype(np.float64)

    return F, H, Q, R


def generate_spiral_curve() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a spiral curve.

    Returns:
        tuple: A tuple containing the x and y coordinates of the spiral curve.
    """
    N = 40
    v = np.linspace(5 * np.pi, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v

    return x, y


def generate_lissaJous_curve() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a Lissajous curve.

    Returns:
        tuple: A tuple containing the x and y coordinates of the Lissajous curve.
    """
    t = np.linspace(0, 2*np.pi, 40)
    A = 1
    B = 2
    a = 3
    b = 4
    x = A * np.sin(a*t)
    y = B * np.sin(b*t)

    return x, y


def generate_butterfly_curve() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a butterfly curve.

    Returns:
        tuple: A tuple containing the x and y coordinates of the butterfly curve.
    """
    t = np.linspace(0, 2*np.pi, 40)
    x = np.sin(t)*(np.exp(np.cos(t)) - 2*np.cos(4*t) - np.sin(t/12)**5)
    y = np.cos(t)*(np.exp(np.cos(t)) - 2*np.cos(4*t) - np.sin(t/12)**5)

    return x, y


def generate_spirograph_curve() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a spirograph curve.

    Returns:
        tuple: A tuple containing the x and y coordinates of the spirograph curve.
    """
    t = np.linspace(0, 2*np.pi, 40)
    R = 5
    r = 3
    d = 1
    x = (R-r)*np.cos(t) + d*np.cos((R-r)/r*t)
    y = (R-r)*np.sin(t) - d*np.sin((R-r)/r*t)

    return x, y


def generate_curves() -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates the x and y coordinates of all functions with the 'generate' prefix.

    Returns:
        list: A list containing the x and y coordinates of the generated curves.
    """
    curves = []

    # Use the 'generate' prefix to identify the functions that generate the curves.
    # Add the generated curves to the list, along with the function name.
    for name in globals():
        if name.startswith('generate_') and name != 'generate_curves':
            x, y = globals()[name]()
            curves.append((name, x, y))

    return curves


if __name__ == "__main__":
    parameters = [(1, 1), (1, 5), (5, 5), (50, 1), (1, 50)]
    motion_models = [random_walk, nearly_constant_velocity, nearly_constant_acceleration]

    fig, ax = plt.subplots(3, 3, figsize=(12, 12))

    for curve in generate_curves():
        name, x, y = curve

        fig, ax = plt.subplots(3, 5, figsize=(15, 15))
    
        for i, motion_model in enumerate(motion_models):
            for j, (q, r) in enumerate(parameters):
                F, H, Q, R = motion_model(q, r)

                ax[i, j].title.set_text(f"{motion_model.__name__}, q: {q}, r: {r}")
                # Set font size for the title.
                ax[i, j].title.set_fontsize(8)

                sx=np.zeros((x.size,1), dtype=np.float32).flatten()
                sy=np.zeros((y.size,1), dtype=np.float32).flatten()
                sx[0]=x[0]
                sy[0]=y[0]

                state=np.zeros((F.shape[0],1), dtype=np.float32).flatten()
                state[0] = x[0]
                state[1] = y[0]

                covariance=np.eye(F.shape[0], dtype=np.float32)
                for k in range(1, x.size):
                    state, covariance, _, _ = kalman_step(F, H, Q, R, np.reshape(
                        np.array([x[k], y[k]]), (-1,1)
                    ), np.reshape(state,(-1,1)), covariance)
                    sx[k]=state[0]
                    sy[k]=state[1]

                ax[i,j].plot(x, y, '-ro')
                ax[i,j].plot(sx, sy, '-bo')

        # Save the plot as a pdf. Make sure previous plots do not show on the current plot.
        plt.savefig(f"./results/motion_models/{name}.pdf")
        #plt.close()



