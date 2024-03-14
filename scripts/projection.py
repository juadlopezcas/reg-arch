from data import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_basis_and_lines(basis_matrix, points):
    # Plot basis axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0, 0, 0, basis_matrix[0][0], basis_matrix[1][0], basis_matrix[2][0], color='r', label='Basis Axis 1')
    ax.quiver(0, 0, 0, basis_matrix[0][1], basis_matrix[1][1], basis_matrix[2][1], color='g', label='Basis Axis 2')
    ax.quiver(0, 0, 0, basis_matrix[0][2], basis_matrix[1][2], basis_matrix[2][2], color='b', label='Basis Axis 3')

    # Plot lines projecting points to closest basis axis
    for point in points:
        closest_axis_index = np.argmax(np.abs(np.dot(point, basis_matrix)))
        closest_axis = basis_matrix[:, closest_axis_index]
        projected_point = np.dot(point, closest_axis) * closest_axis
        ax.plot([point[0], projected_point[0]], [point[1], projected_point[1]], [point[2], projected_point[2]], 'k--')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Projection onto Closest Orthonormal Basis Axis')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    gen_dat = DataGenerator(1000, 3)
    matrix = gen_dat.matrix
    points = gen_dat.cross_data()
    # Plot basis axes and lines projecting points to closest basis axis
    plot_basis_and_lines(matrix, points)
