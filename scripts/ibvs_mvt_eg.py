#!/usr/bin/env python3

import machinevisiontoolbox as mv
import matplotlib.pyplot as plt
import spatialmath as sm
import numpy as np


def projection_exp(cam, X, Y, Z):
    fx = cam.fu
    fy = cam.fv
    cx = cam.width // 2
    cy = cam.height // 2
    rho_u = cam.rhou
    rho_v = cam.rhov

    x_ = fx * (X / Z)
    y_ = fy * (Y / Z)
    pix_x = x_ / rho_u + cx
    pix_y = y_ / rho_v + cy

    print(pix_x, pix_y)

def main():
    # camera model
    cam = mv.CentralCamera().Default()

    print(cam.project_point([0.2, 0.3, 1]))
    projection_exp(cam, 0.2, 0.3, 1)

    point = np.array([[500], [500]])
    cam.plot_point(point)

    point = np.array([[0], [500]])
    cam.plot_point(point)

    point = np.array([[300], [500]])
    cam.plot_point(point)

    plt.show()

if __name__ == "__main__":
    main()