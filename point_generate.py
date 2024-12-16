import numpy as np
import matplotlib.pyplot as plt
import random


def generate_cube_point(side_length=0.1, num_samples=1000):
    sampled_points = np.zeros((num_samples, 6))
    max_degree = np.deg2rad(30)

    for i in range(num_samples):

        x = np.random.uniform(-side_length / 2, side_length / 2)
        y = np.random.uniform(-side_length / 2, side_length / 2)
        z = np.random.uniform(-side_length / 2, side_length / 2)

        phi = max_degree * random.uniform(-1, 1)
        theta = max_degree * random.uniform(-1, 1)
        psi = max_degree * random.uniform(-1, 1)
        sampled_points[i] = [x, y, z, phi, theta, psi]

    return sampled_points


def draw_scatter(points):
    fig = plt.figure(dpi=200, constrained_layout=True)
    plt.rcParams['font.family'] = 'Times New Roman'
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontweight('bold')
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontweight('bold')
    for tick in ax.zaxis.get_majorticklabels():
        tick.set_fontweight('bold')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

    ax.set_xlim(-0.06, 0.06)
    ax.set_ylim(-0.06, 0.06)
    ax.set_zlim(-0.06, 0.06)

    plt.show()


if __name__ == "__main__":
    points = generate_cube_point(side_length=0.12, num_samples=1000)
    draw_scatter(points=points)
