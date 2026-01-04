#!/usr/bin/env python3

import machinevisiontoolbox as mv
import matplotlib.pyplot as plt
import spatialmath as sm
import numpy as np

def main():
    # camera model
    cam = mv.CentralCamera().Default()
    print(f"Camera default parameters: {cam}")

    d = 0.5
    z = 1
    lambda_ = 0.1   # arbitrary value
    
    # these are current feature points in world frame (i guess)
    p1 = [d, -(np.sqrt(3)/4) * d, z]
    p2 = [-d, -(np.sqrt(3)/4) * d, z]
    p3 = [0, (np.sqrt(3)/4) * d, z]
    P = [p1, p2, p3]

    # desired image frame points
    D_P = [[250, 500, 750], [750, 250, 750]]


    # project and plot the points
    current_points = cam.plot_point(P, pose=sm.SE3(0,0,-5)*sm.SE3.Ry(10, unit='deg'))
    target_points = cam.plot_point(D_P, 'r*')
    # logs
    print(f"Current image points: {current_points}")
    print(f"Desired image points: {target_points}")


    # image jacobian (for all three points)
    img_jac = cam.visjac_p(current_points, 5)       # we are setting the current point is 5m away from world points (WIP)
    inv_img_jac = np.linalg.inv(img_jac)
    print(f"Image Jacobian Matrix: \n{img_jac}\n")
    print(f"Inv Image Jacobian Matrix: \n{inv_img_jac}\n")

    # compute image pixel velocity
    target_pixel_velocities = lambda_ * (target_points - current_points)
    print(f"Target Pixel Velocity: \n{target_pixel_velocities}\n")


    # compute camera velocity using image jacobian
    temp_pix_vel = np.array(target_pixel_velocities).T.flatten()
    temp_pix_vel = np.array([temp_pix_vel]).T
    cam_vel = np.matmul(inv_img_jac, temp_pix_vel)
    print(f"Camera Velocities: \n{cam_vel}\n")


    # next steps:
    # 1. move the camera or change pose of camera based on velocity ouput from the above
    # 2. the controller should minimize the output velocity
    # 3. less than 0.001 or lesser velocity generated should be considered as converged

    plt.show()

if __name__ == "__main__":
    print('\n', end='')     # just for better visibility
    main()