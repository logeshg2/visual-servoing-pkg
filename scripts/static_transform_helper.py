#!/usr/bin/env python3

"""
This is a helper script to compute static transform between two known transforms.
"""

import numpy as np
from scipy.spatial.transform import Rotation


if __name__ == "__main__":

    bTe1 = np.eye(4)
    bTe1[0:3, 3] = np.array([154.70700073242188, 255.8056182861328, -33.64824676513672]) / 1000
    bTe1[0:3, 0:3] = Rotation.from_euler("xyz", [-179.65093994140625, 0.6298328638076782, 93.67679595947266], degrees=True).as_matrix()
    bTe2 = np.eye(4)
    bTe2[0:3, 3] = np.array([131.49952697753906, 318.28564453125, -116.39675903320312]) / 1000
    bTe2[0:3, 0:3] = Rotation.from_euler("xyz", [-179.55535888671875, 0.7517288327217102, 96.01161193847656], degrees=True).as_matrix()
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