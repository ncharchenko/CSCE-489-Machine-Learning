import numpy as np
'''
Homework5: Principal Component Analysis

Helper functions
----------------
In this assignment, you may want to use following helper functions:
- np.linalg.svd(): compute the singular value decomposition on a given matrix. 
- np.dot(): matrices multiplication operation.
- np.mean(): compute the mean value on a given matrix.
- np.ones(): generate a all '1' matrix with a given shape.
- np.transpose(): matrix transpose operation.
- np.linalg.norm(): compute the norm value of a matrix.

'''

class PCA():

	def __init__(self, X, n_components):
		'''
		Args:
			X: The data matrix of shape [n_samples, n_features].
			n_components: The number of principal components. A scaler number.
		'''

		self.n_components = n_components
		self.X = X
		self.Up, self.Xp = self._do_pca()

	
	def _do_pca(self):
		'''
		To do PCA decomposition.
		Returns:
			Up: Principal components (transform matrix) of shape [n_features, n_components].
			Xp: The reduced data matrix after PCA of shape [n_samples, n_components].
		'''
		X_t = self.X.T
		self.mean = np.mean(X_t,axis=1)
		self.mat = self.X -self.mean
		u,s,v = np.linalg.svd(self.mat)
		self.Up = u[:,:self.n_components]
		self.Xp = np.dot(self.Up.T,self.mat)
		print(self.Xp.shape)
		return(self.Up,self.Xp)

	def get_reduced(self, X=None):
		'''
		To return the reduced data matrix.
		Args:
			X: The data matrix with shape [n_any, n_features] or None. 
			   If None, return reduced training X.
		Returns:
			Xp: The reduced data matrix of shape [n_any, n_components].
		'''
		if X is None:
			return self.Xp
		else:
			return X@self.Up

	def reconstruction(self, Xp):
		'''
		To reconstruct reduced data given principal components Up.

		Args:
		Xp: The reduced data matrix after PCA of shape [n_samples, n_components].

		Return:
		X_re: The reconstructed matrix of shape [n_samples, n_features].
		'''
		X_re = np.dot(self.Up,self.Xp)
		X_re = X_re + self.mean
		return X_re


def reconstruct_error(A, B):
	'''
	To compute the reconstruction error.

	Args: 
	A & B: Two matrices needed to be compared with. Should be of same shape.

	Return: 
	error: the Frobenius norm's square of the matrix A-B. A scaler number.
	'''
	return np.linalg.norm(A-B, ord='fro')
	

