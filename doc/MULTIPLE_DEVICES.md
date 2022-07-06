# Dynamic-VINS multiple devices running procedure

For example,
Intel NUC as the master device and HUAWEI Atlas200DK as the edge device.

1. first, find devices IP respectively.
   ```bash
   sudo apt install net-tools
   ifconfig
   ```
   take the master device (Intel NUC) IP as `192.168.2.223` and the edge device (HUAWEI Atlas200DK) IP as `192.168.2.2`.


2. ```bash
   # run vins on the master device (Intel NUC)
   export ROS_HOSTNAME=192.168.2.223
   export ROS_MASTER_URI=http://192.168.2.223:11311
   roscore
   ```

4. (optional) compressed color images.
   
   This step is required as the limited bandwidth.

   If your image topic with compressed image topic originally, this step is not necessary.
   ```bash
   # run vins on the master device (Intel NUC)
   export ROS_HOSTNAME=192.168.2.223
   export ROS_MASTER_URI=http://192.168.2.223:11311
   rosrun image_transport republish raw in:=/d400/color/image_raw compressed out:=/d400/color/image_raw
   ```

5. **Dynamic-VINS** on the edge device (HUAWEI Atlas200DK)

      ```bash
         # run vins on the edge device (HUAWEI Atlas200DK)
         ssh HwHiAiUser@192.168.2.2
         export ROS_HOSTNAME=192.168.2.2
         export ROS_MASTER_URI=http://192.168.2.223:11311
         roslaunch vins_estimator openloris_vio_atlas.launch
      ```

6. rviz visualiztion
   ```
   # run vins on the master device (Intel NUC)
   export ROS_HOSTNAME=192.168.2.223     #从机IP 
   export ROS_MASTER_URI=http://192.168.2.223:11311
   roslaunch vins_estimator vins_rviz.launch 
   ```
