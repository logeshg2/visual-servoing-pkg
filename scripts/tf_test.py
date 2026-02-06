#!/usr/bin/env python3


import pickle
import numpy as np
import spatialmath as sm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from fanuc_vel_controller import fanuc_model


mat = pickle.load(open("/home/logesh/fanuc_ws/src/visual-servoing-pkg/config/eye_in_hand_rs.pkl", "rb"))
rotm = Rotation.from_matrix(mat[0:3, 0:3]).as_matrix()
eTc = np.eye(4)
eTc[0:3, 3] = mat[0:3, 3]
eTc[0:3, 0:3] = rotm
eTc = sm.SE3(eTc)

# plotting
# sm.SE3().plot(length=0.1, frame='O')
# eTc.plot(length=0.1, color='green', frame='A')


# plt.show()


tar_pose = sm.SE3(0.459, 0.034, 0.118)
rotm = np.array([
    [0.022, -1.000,  0.012], 
    [-1.000,  0.023,  0.007], 
    [-0.007, -0.012, -1.000]
])
tar_pose.R = Rotation.from_matrix(rotm).as_matrix()

model = fanuc_model.Fanuc()
q_tar = model.ik_LM(tar_pose)[0]
model.plot(q_tar, block=True)
plt.show()