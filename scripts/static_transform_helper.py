#!/usr/bin/env python3

"""
This is a helper script to compute static transform between two known transforms.
"""

import numpy as np
from scipy.spatial.transform import Rotation


if __name__ == "__main__":

    bTe1 = np.eye(4)
    bTe1[0:3, 3] = np.array([126.13108825683594, 242.6962432861328, -34.67491149902344]) / 1000
    bTe1[0:3, 0:3] = Rotation.from_euler("xyz", [179.2032470703125, 0.17680034041404724, 81.62097930908203], degrees=True).as_matrix()
    bTe2 = np.eye(4)
    bTe2[0:3, 3] = np.array([115.81856536865234, 307.5487976074219, -111.48181915283203]) / 1000
    bTe2[0:3, 0:3] = Rotation.from_euler("xyz", [179.20362854003906, 0.1783638745546341, 81.62091827392578], degrees=True).as_matrix()
    e1Te2 = np.linalg.inv(bTe1) @ bTe2

    print()
    print("Static transform between e1 and e2:")
    print(e1Te2)
    print()
    print("Translation (in m)")
    print(e1Te2[0:3, 3].flatten())
    print()
    print("Rotation (matric)")
    print(e1Te2[0:3, 0:3])
    print()
    print("Euler Angles (xyz) - deg")
    print(Rotation.from_matrix(e1Te2[0:3, 0:3]).as_euler("xyz", degrees=True))
    print()