#!/usr/bin/env python3

import machinevisiontoolbox as mv
import matplotlib.pyplot as plt
import spatialmath as sm
import numpy as np


DEBUG = True

# camera model
cam = mv.CentralCamera().Default()
print(f"Camera default parameters: {cam}")

d = 0.5
z = 5
lambda_ = 0.05   # arbitrary value

# these are current feature points in world frame (i guess)
p1 = [d, -(np.sqrt(3)/4) * d, z]
p2 = [-d, -(np.sqrt(3)/4) * d, z]
p3 = [0, (np.sqrt(3)/4) * d, z]
P = [p1, p2, p3]

# desired image frame points
D_P = [[250, 500, 750], [750, 250, 750]]


def projectPoints(cam_pose: sm.SE3):
    """
    Function to plot the point's (in world frame).
    And also plot the desired point's (in the image frame).
    
    args:
        cam_pose: camera pose in which world points are viewed

    return:
        current_points: pixel coordinates of world points
        target_points: pixel coordinates of target points
    """

    # project and plot the points
    current_points = cam.plot_point(P, pose=cam_pose)
    target_points = cam.plot_point(D_P, 'r*')
    if (DEBUG):
        # logs
        print(f"Current image points: {current_points}")
        print(f"Desired image points: {target_points}")

    return current_points, target_points


def computeCamVel(inv_img_jac, current_points, target_points):
    """
    Function to compute desired camera velocity from the current and desired pixel points.
    CamVel = inv(imgJac) * (pixel_vels)
    
    args:
        current_points: pixel coordinates of world points
        target_points: pixel coordinates of target points

    return:
        camVel: desired velocity in camera coordinate frame
    """

    target_pixel_velocities = lambda_ * (target_points - current_points)
    if (DEBUG):
        print(f"Target Pixel Velocity: \n{target_pixel_velocities}\n")

    # compute camera velocity using image jacobian
    temp_pix_vel = np.array(target_pixel_velocities).T.flatten()
    temp_pix_vel = np.array([temp_pix_vel]).T
    cam_vel = np.matmul(inv_img_jac, temp_pix_vel)
    if (DEBUG):
        print(f"Camera Velocities: \n{cam_vel}\n")

    return cam_vel


def plotting_function(cpose):
    cam.plot(pose=cpose)
    
    # world points
    sm.base.plot_point(p1)
    sm.base.plot_point(p2)
    sm.base.plot_point(p3)

    plt.pause(0.05)
    plt.show(block=False)
    plt.cla()

def main():
    
    # camera pose (this is initial)
    CPose = sm.SE3(0,0,-4)
    
    # main loop
    # compute convergence from the resultant velocity (check if it is close to zero)
    try:
        while True: 
            # 1. project the world and target points with respect to the camera
            current_points, target_points = projectPoints(cam_pose = CPose)

            # 2. compute image jacobian (for all three points) - for 3 points ~> inv(Jcam) is (6, 6) matrix
            # need inv of jacobian for pixel velocity to camera velocity conversion
            img_jac = cam.visjac_p(current_points, 2)       # we are setting the current point is 5m away from world points (WIP)
            inv_img_jac = np.linalg.inv(img_jac)
            if (DEBUG):
                print(f"Image Jacobian Matrix: \n{img_jac}\n")
                print(f"Inv Image Jacobian Matrix: \n{inv_img_jac}\n")

            # 3. compute image pixel velocity
            cam_vel = computeCamVel(inv_img_jac, current_points, target_points)

            # 4. velocity control (like) - need to work on this
            cam_vel *= 0.005
            CPose.x += cam_vel[0]
            CPose.y += cam_vel[1]         
            CPose.z += cam_vel[2]
            # CPose = CPose @ sm.SE3.Rx(theta=cam_vel[3])
            # CPose = CPose @ sm.SE3.Ry(theta=cam_vel[4])
            # CPose = CPose @ sm.SE3.Rz(theta=cam_vel[5])
    
            # plotting function
            plotting_function(CPose)
    except Exception as e:
        print(f"Exception: {e}")
        plt.close('all')
        exit(0)

if __name__ == "__main__":
    print('\n', end='')     # just for better visibility
    main()