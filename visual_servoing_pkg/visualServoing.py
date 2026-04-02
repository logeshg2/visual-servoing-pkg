#!/usr/bin/env python3

"""
The main pkg for visual servoing using python, inspired from VISP team - based servoing functionalities.
Currently implemented for Image based visual servoing approach.
"""


import numpy as np
from scipy.spatial.transform import Rotation



class VisualServoingPkg():
    def __init__(
            self,
            vsMode: str="ibvs",                         # ['ibvs' or 'pbvs']
            fixedPoint: bool=True,
        ):

        # handle parameters
        assert (vsMode.lower() in {'ibvs', 'pbvs'}), "Incompatible servoing mode - ['ibvs' or 'pbvs']"
        

        # basic variables
        self.vsMode = vsMode









if __name__ == "__main__":
    pass