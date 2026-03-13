# Visual - Servoing - Package

Just trying out visual servoing - without using any third part visual servoing pkg - creating my own...

Working branch: plug_pick_insert_bt

---

### Implementation Details:

The repository depends on <a href="https://github.com/logeshg2/fanuc_ros2_drivers" target="_blank">Fanuc ROS2 Driver</a> package for read robot control.

The implementation of this visual servoing packages was done using the reference from the following papers:

1. Visual servo control, Part I: Basic approaches: <a href="https://www.irisa.fr/lagadic/pdf/2006_ieee_ram_chaumette.pdf" target="_blank">Link</a>

2. Other visual servoing papers can be found here: <a href="https://visp.inria.fr/publications/" target="_blank">Inria - ViSP</a>

---

### Repo Details:

1. <a href="./config/" target="_blank">Config</a>: Contains calibrated transforms for camera (intrinsic and extrinsic).
    
    Here are some resource to perform calibration:
        
    1. <a href="https://github.com/logeshg2/Camera-Calibration" target="_target">Camera Intrinsic</a> 

    2. <a href="https://github.com/logeshg2/moveit2_calibration" target="_target">Camera Extrinsic</a>

2. <a href="./scripts/" target="_blank">Scripts</a>: Contains ROS2 vision nodes for image acquisition and processing for different objects like Aruco, Extension Box, Plug Holder. It also contains some other testing py files.

    Simple Object Pose using YOLO and PnP can be found here:  <a href="https://github.com/logeshg2/ObjectPose-simple" target="_target">ObjectPose-simple</a>

3. <a href="./visual_servoing_pkg/" target="_blank">visual_servoing_pkg</a>: Contains ROS2 visual servoing nodes for different approachs like **IBVS**, and **PBVS**. The servoing node can be found for different objects like aruco, n-points, and extension box.

4. <a href="./vs_bt/" target="_blank">vs_bt</a>: Contains Py-Trees implemenation for visual servoing + pick and place task sequence. It contains difference behaviours like aruco servoing, socket servoing, insertion subtree, and picking substree.

    - <a href="./vs_bt/main_tree.py" target="_blank">main_tree.py</a> has the logic to connect different behaviours to form tree to perform sequential execution.
    - <a href="./visual_servoing_pkg/visualServoing_BT.py" target="_blank">visualServoing_BT.py</a> has the background logic to perform high frequency visual servoing. This is a main ROS2 node that communicates with behaviours in the py-trees and commands the real robotic arm.

---

### Visual Servoing Plus Assembly Task (Demo):

[![Watch the video](https://img.youtube.com/vi/lK--YVCa8BM/0.jpg)](https://www.youtube.com/watch?v=lK--YVCa8BM)