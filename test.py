import numpy as np
from scipy.linalg import block_diag

Q = np.diag([1,2,3])

R = np.diag([4,5,6])

result = block_diag(Q,R)
print(result)


