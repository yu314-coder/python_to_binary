import numpy as np

matrix = np.array([[1, 2], [3, 4]])
print("NumPy", np.__version__, "determinant", round(float(np.linalg.det(matrix))))
