import numpy as np
from sklearn.preprocessing import normalize

#x = np.random.rand(1000)*10
#x = np.array([119,126,120,116,173,153,137])
#x = np.array([3,10,4,0,57,37,21])
#x = np.array([4,11,5,1,58,38,22])
x = np.array([0.0135395, 0.054158, 0.0676975, 0.1489345, 0.297869, 0.514501, 0.82329099])

#x = np.array([0.0135395, 0.054158, 0.0676975, 0.1489345, 0.297869, 0.514501, 0.78529099]) # 0,6
#x = np.array([0.0135395, 0.054158, 0.0676975, 0.1489345, 0.297869, # O,4
#x = np.array([0.0135395, 0.054158, 0.0676975, 0.1489345,  # O,2



norm1 = x / np.linalg.norm(x)
norm2 = normalize(x[:,np.newaxis], axis=0).ravel()
norm2 = np.sort(norm2)
print (norm1)
print (norm2)

#print (np.all(norm1 == norm2))
# True