# import xrobotoolkit_sdk as xrt
# xrt.init()

# while True:
#     if xrt.get_A_button():
#         print("A button pressed")
#         break
# xrt.close()

from Robotic_Arm.rm_robot_interface import *

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("192.168.10.19", 8080)
print(handle.id)
# 灵巧手角度跟随控制
# print(arm.rm_set_hand_posture(1, True, 10))
print(arm.rm_set_hand_angle([100,100,100,100,100,100],True,1))
print(arm.rm_set_hand_angle([2000,2000,2000,2000,2000,2000],True,1))
arm.rm_delete_robot_arm()