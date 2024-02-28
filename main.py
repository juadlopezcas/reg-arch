import autograd.numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pymanopt as manopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers
from pymanopt.tools.diagnostics import check_gradient
from data import DataGenerator
from scripts.plots import plot_gaussians_with_axes

#SUPPORTED_BACKENDS = ("autograd", "numpy", "pytorch")
manifold = manifolds.SpecialOrthogonalGroup(3)
@manopt.function.autograd(manifold)
def cost(basis_matrix):
    np.random.seed(0)
    dist = []
    gen_dat = DataGenerator(1000, 3)
    points = gen_dat.cross_data()
    for point in points:
        closest_axis_index = np.argmax(np.abs(np.dot(point, basis_matrix)))
        closest_axis = basis_matrix[:, closest_axis_index]
        projected_point = np.dot(point, closest_axis) * closest_axis
        squared_distance = np.linalg.norm((point - projected_point)**2)
        dist.append(squared_distance)
    return np.trapz(dist)

problem = manopt.Problem(manifold=manifold, cost=cost)

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    gen_dat = DataGenerator(1000, 3)
    print(gen_dat.matrix)
    points = gen_dat.cross_data()
    plot_gaussians_with_axes(gen_dat.matrix, points, title='Original orthogonal axis and data points')
    solver = optimizers.SteepestDescent(verbosity=1)
    x_opt = solver.run(problem).point
    print(x_opt)
    plot_gaussians_with_axes(x_opt, points, title='Optimized axis and data points')
    
