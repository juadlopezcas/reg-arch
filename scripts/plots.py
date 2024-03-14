from data import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def plot_PCA_WFA(matrix,samples):
    # Perform PCA on the samples
    if matrix.shape[0] == 2:
        pca = PCA(n_components=2)
        pca.fit(samples)

        # Get the mean of the data, principal components, and their explained variance
        mean = pca.mean_
        components = pca.components_
        explained_variance = pca.explained_variance_

        # Plot the original data
        plt.figure(figsize=(8, 6))
        plt.scatter(samples[:, 0],samples[:, 1], alpha=0.5, label='Data', color = 'slateblue' )

        # Plot the principal component vectors
        v1 = components[0] * 3 * np.sqrt(explained_variance[0])  # Scale the vectors for visualization
        v2 = components[1] * 3 * np.sqrt(explained_variance[1])  # Scale the vectors for visualization
        plt.quiver(mean[0], mean[1], v1[0], v1[1], angles='xy', scale_units='xy', scale=1, width=0.009, color='tomato', label='PCA', zorder=5)
        plt.quiver(mean[0], mean[1], v2[0], v2[1], angles='xy', scale_units='xy', scale=1, width=0.009, color='tomato', zorder=5)
    
        #for i, (variance, vector) in enumerate(zip(explained_variance, components)):
        #    v = vector * 3 * np.sqrt(variance)  # Scale the vectors for visualization
        #    plt.quiver(mean[0], mean[1], v[0], v[1], angles='xy', scale_units='xy', scale=1, width=0.005, color='tomato', label='PCA', zorder=5)

        plt.quiver(0, 0, 3 * np.sqrt(explained_variance[0])*matrix[0, 0], 3 * np.sqrt(explained_variance[0])*matrix[1, 0], angles='xy', scale_units='xy', scale=1, width=0.009, color='yellowgreen', label='Wasserstein Factor Analysis')
        plt.quiver(0, 0, 3 * np.sqrt(explained_variance[0])*matrix[0, 1], 3 * np.sqrt(explained_variance[0])*matrix[1, 1], angles='xy', scale_units='xy', scale=1, width=0.009, color='yellowgreen')
        #plt.title('PCA on Mixture of Gaussians with Covariance 0.92')
        #plt.xlabel('x')
        #plt.ylabel('X2')
        plt.legend(fontsize="12")
        #plt.grid(True)
        plt.show()
    if matrix.shape[0] == 3:
        pca = PCA(n_components=3)
        pca.fit(samples)

        # Get the mean of the data, principal components, and their explained variance
        mean = pca.mean_
        components = pca.components_
        explained_variance = pca.explained_variance_

        # Plot the original data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.5, label='Data', color = 'slateblue')

        # Plot the principal component vectors
        v1 = components[0] * 3 * np.sqrt(explained_variance[0]) # Scale the vectors for visualization
        v2 = components[1] * 3 * np.sqrt(explained_variance[1]) # Scale the vectors for visualization
        v3 = components[2] * 3 * np.sqrt(explained_variance[2]) # Scale the vectors for visualization
        ax.quiver(mean[0], mean[1], mean[2], v1[0], v1[1], v1[2], color='tomato', label='PCA', zorder=5)
        ax.quiver(mean[0], mean[1], mean[2], v2[0], v2[1], v2[2], color='tomato', zorder=5)
        ax.quiver(mean[0], mean[1], mean[2], v3[0], v3[1], v3[2], color='tomato', zorder=5)

        ax.quiver(0, 0, 0, 3 * np.sqrt(explained_variance[0])*matrix[0, 0], 3 * np.sqrt(explained_variance[0])*matrix[1, 0], 3 * np.sqrt(explained_variance[0])*matrix[2, 0], color='yellowgreen', label='Wasserstein Factor Analysis')
        ax.quiver(0, 0, 0, 3 * np.sqrt(explained_variance[0])*matrix[0, 1], 3 * np.sqrt(explained_variance[0])*matrix[1, 1], 3 * np.sqrt(explained_variance[0])*matrix[2, 1], color='yellowgreen')
        ax.quiver(0, 0, 0, 3 * np.sqrt(explained_variance[0])*matrix[0, 2], 3 * np.sqrt(explained_variance[0])*matrix[1, 2], 3 * np.sqrt(explained_variance[0])*matrix[2, 2], color='yellowgreen')
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        #ax.set_title('PCA on Mixture of Gaussians with Covariance 0.92')
        ax.legend(fontsize="12")
        #plt.grid(True)
        plt.show()

def plot_gaussians_with_axes(matrix, data, title = 'Gaussian data with axes'):
    if matrix.shape[0] == 2: 
        fig, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], color='y', alpha=0.5)
        ax.quiver(0, 0, matrix[0, 0], matrix[1, 0], angles='xy', scale_units='xy', scale=1, color='b')
        ax.quiver(0, 0, matrix[0, 1], matrix[1, 1], angles='xy', scale_units='xy', scale=1, color='g')
        ax.set_aspect('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
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
        ax.set_title(title)
        plt.show()
    else:
        print("Invalid matrix dimensions. Supported dimensions are 2x2 and 3x3.")



if __name__ == "__main__":
    gen_dat = DataGenerator(5000, 3)
    gaussian_samples = gen_dat.cross_data()
    plot_gaussians_with_axes(gen_dat.matrix, gaussian_samples)
