import numpy as np

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