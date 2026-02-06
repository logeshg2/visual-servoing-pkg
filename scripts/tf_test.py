#!/usr/bin/env python3


import pickle
import numpy as np
import spatialmath as sm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation



mat = pickle.load(open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/eye_in_hand_rs.pkl", "rb"))
rotm = Rotation.from_matrix(mat[0:3, 0:3]).as_matrix()
eTc = np.eye(4)
eTc[0:3, 3] = mat[0:3, 3]
eTc[0:3, 0:3] = rotm
eTc = sm.SE3(eTc)

# plotting
sm.SE3().plot(length=0.1, frame='O')
eTc.plot(length=0.1, color='green', frame='A')


plt.show()