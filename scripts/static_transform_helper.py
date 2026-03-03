#!/usr/bin/env python3

"""
This is a helper script to compute static transform between two known transforms.
"""

import numpy as np
from scipy.spatial.transform import Rotation


if __name__ == "__main__":

    bTe1 = np.eye(4)
    bTe1[0:3, 3] = np.array([49.712303161621094, 283.6398010253906, -5.360454559326172]) / 1000
    bTe1[0:3, 0:3] = Rotation.from_euler("xyz", [-179.99005126953125, 0.4922199845314026, 94.33748626708984], degrees=True).as_matrix()
    bTe2 = np.eye(4)
    bTe2[0:3, 3] = np.array([9.999263763427734, 352.8124084472656, -103.57505798339844]) / 1000
    bTe2[0:3, 0:3] = Rotation.from_euler("xyz", [-179.99005126953125, 0.502923846244812, 94.3348617553711], degrees=True).as_matrix()
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