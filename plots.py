from data import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_gaussians_with_axes(matrix, data):
    if matrix.shape[0] == 2: 
        fig, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], color='y', alpha=0.5)
        ax.quiver(0, 0, matrix[0, 0], matrix[1, 0], angles='xy', scale_units='xy', scale=1, color='b')
        ax.quiver(0, 0, matrix[0, 1], matrix[1, 1], angles='xy', scale_units='xy', scale=1, color='g')
        ax.set_aspect('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Gaussians with Axes')
        plt.grid()
        plt.show()
    elif matrix.shape[0] == 3: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='y',alpha=0.5)
        origin = [0, 0, 0]
        ax.quiver(origin[0], origin[1], origin[2], matrix[0, 0], matrix[1, 0], matrix[2, 0], color='b')
        ax.quiver(origin[0], origin[1], origin[2], matrix[0, 1], matrix[1, 1], matrix[2, 1], color='g')
        ax.quiver(origin[0], origin[1], origin[2], matrix[0, 2], matrix[1, 2], matrix[2, 2], color='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Gaussians with Axes')
        plt.show()
    else:
        print("Invalid matrix dimensions. Supported dimensions are 2x2 and 3x3.")


if __name__ == "__main__":
    gen_dat = DataGenerator(5000, 3)
    gaussian_samples = gen_dat.cross_data()
    plot_gaussians_with_axes(gen_dat.matrix, gaussian_samples)