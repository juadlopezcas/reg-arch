import numpy as np
from sklearn.preprocessing import scale
from scipy.stats import expon


class DataGenerator:
    def __init__(self, num_samples, num_dimensions):
        self.num_samples = num_samples
        self.num_dimensions = num_dimensions
        self.matrix = self.generate_orthonormal_matrix()
    
    def generate_orthonormal_matrix(self):
        return np.linalg.qr(np.random.randn(self.num_dimensions, self.num_dimensions))[0]
    
    def cross_data(self,):
        orthogonal_matrix = self.matrix
        means = np.random.normal(0, 0.25,self.num_dimensions)
        covariances = np.diag(np.random.lognormal(0, 0.5, self.num_dimensions))
        samples = np.random.multivariate_normal(means, covariances, self.num_samples)
        m, n = samples.shape
        projected_data = []
        for i in range(n):
            column_vector = orthogonal_matrix[:, i]
            for j in range(m):
                projection_vector = np.dot(samples[j], column_vector)*column_vector
                projected_data.append(projection_vector)
        return np.concatenate((np.array(projected_data),0.6*samples))
    
    def gaussian_factor_analysis_data(self,):
        np.random.seed(5)
        if self.num_dimensions == 2:
            # Parameters for the Gaussian components
            n_samples = 1000
            u = 0.92
            covariances = [np.array([[1, u], [u, 1]]), np.array([[1, -u], [-u, 1]])]
            weights = [0.5, 0.5]  # Mixing probabilities

            # Generate samples
            samples = np.vstack([
                np.random.multivariate_normal(mean=[0, 0], cov=covariances[0], size=int(n_samples * weights[0])),
                np.random.multivariate_normal(mean=[0, 0], cov=covariances[1], size=int(n_samples * weights[1]))
            ])

            # Shuffle the samples to mix the two distributions
            np.random.shuffle(samples)
        elif self.num_dimensions == 3:
            # Parameters for the Gaussian components
            n_samples = 1000
            
            '''
            u = 0.92
            covariances = [np.array([[1, u, -u], [u, 1, u], [-u, u, 1]]), np.array([[-u, 1, -u], [-1, u, u], [-u, -u, 1]]), np.array([[-1, -u, -u], [-u, -1, u], [u, -1, 1]])]
            weights = [0.5, 0.5, 0.5]
            # Generate samples
            samples = np.vstack([
                np.random.multivariate_normal(mean=np.zeros(3), cov=covariances[0], size=int(n_samples * weights[0])),
                np.random.multivariate_normal(mean=np.zeros(3), cov=covariances[1], size=int(n_samples * weights[1])),
                np.random.multivariate_normal(mean=np.zeros(3), cov=covariances[2], size=int(n_samples * weights[2]))
            ])
            '''
            num_petals = 3  # Tetrahedral symmetry
            num_points_per_petal = 100
            petal_length = 1.0
            petal_width = 0.2

            # Generate points for each petal
            theta = np.linspace(0, 2 * np.pi, num_points_per_petal)
            phi = np.linspace(0, 2 * np.pi, num_petals + 1)[:-1]
            theta, phi = np.meshgrid(theta, phi)
            r = petal_length * (1 + petal_width * np.sin(num_petals * theta))

            x = r * np.cos(theta) * np.sin(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(phi)

            # Flatten arrays to plot
            x = x.flatten()
            y = y.flatten()
            z = z.flatten()

            # Create points inside the petals with normal distribution along three planes
            num_points_inside = 2000
            petal_angles = np.linspace(0, 2 * np.pi, num_petals + 1)[:-1]
            petal_centers = np.array([[np.cos(angle), np.sin(angle), 0] for angle in petal_angles])

            samples = np.empty((0, 3))
            for center in petal_centers:
                # Generate random points in three orthogonal planes
                random_points = np.random.normal(size=(num_points_inside, 3))
                random_points -= np.dot(random_points, center)[:, None] * center
                random_points = np.dot(random_points, np.linalg.qr(np.random.randn(3,3))[0]) # Rotate points randomly
                # Scale to fit inside the petal
                random_points *= np.random.uniform(0, petal_length, size=(num_points_inside, 1))
                samples = np.vstack((samples, random_points))
            # Shuffle the samples to mix the two distributions
            np.random.shuffle(samples)
        else:
            raise ValueError('Number of dimensions must be 2 or 3')
        return samples
    
    def exponential_factor_analysis_data(self,):
        np.random.seed(5)
        if self.num_dimensions == 2:
            # Parameters for the exponential components
            n = self.num_samples
            norm_dat = np.random.normal(size=(n, 2))
            norm_dat = scale(norm_dat)

            exp_dat = np.random.choice([-1, 1], size=(2 * n, 2)) * expon.rvs(size=(2 * n, 2)) ** 1.3
            exp_dat = scale(exp_dat)

            s = np.linalg.svd([[1, -2], [-3, 1]])
            R = s[0]
            rexp_dat = np.dot(exp_dat, R)
        if self.num_dimensions == 3:
            # Parameters for the exponential components
            n = self.num_samples
            norm_dat = np.random.normal(size=(n, 3))
            norm_dat = scale(norm_dat)

            exp_dat = np.random.choice([-1, 1], size=(2 * n, 3)) * expon.rvs(size=(2 * n, 3)) ** 1.3
            exp_dat = scale(exp_dat)

            s = np.linalg.svd([[1, -2, 1], [-3, 1, 1], [1, 1, -1]])
            R = s[0]
            rexp_dat = np.dot(exp_dat, R)
        return rexp_dat
