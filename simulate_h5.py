# import gymnasium as gym
# import yumi_gym
import gym, yumi_gym
import pybullet as p
import numpy as np
import h5py
import time
# from Robotic_Arm.rm_robot_interface import *
import math
# import keyboard
def trans2realworld(angle,min,max):
    '''
    要转换为角度,且检查是否超限,输入为弧度,下限,上限
    '''
    #检查长度是否一致
    if len(angle)!=len(min) or len(angle)!=len(max):
        print('长度不一致')
        return
    # 将 angle 转换为 numpy 数组以支持数值运算
    angle_np = np.array(angle)
    angle_real = angle_np * 180.0 / math.pi
    # 使用 enumerate 来获取索引和值
    for i, ang in enumerate(angle_real):
        if ang > max[i]:
            angle_real[i] = max[i] - 1
            print('角度过大')
        elif ang < min[i]:
            angle_real[i] = min[i] + 1
            print('角度过小')
    
    #转为list
    angle_real = angle_real.tolist()
    print(angle_real)
    return angle_real

# C:\Users\30687\Desktop\ur3_robot_train_pico\ur3_robot_train_pico\saved\h5\测试-ceshi_1800.h5
hf = h5py.File("C:\\Users\\30687\\Desktop\\ur3_robot_train_pico\\ur3_robot_train_pico\\saved\\h5\\测试-ceshi_1800.h5", 'r') #朝向

group = hf.get('group1')

l_joint_angle = group.get('l_joint_angle')
r_joint_angle = group.get('r_joint_angle')
print('左手动作',l_joint_angle)
l_joint_angle_np = np.array(l_joint_angle)
r_joint_angle_np = np.array(r_joint_angle)
print('左手手臂',l_joint_angle_np.shape)
print('右手手臂',r_joint_angle_np.shape)
# l_hand_angle = group.get('l_glove_angle')
# r_hand_angle = group.get('r_glove_angle')

# l_glove_angle_np = np.array(l_hand_angle)
# r_glove_angle_np = np.array(r_hand_angle)
# print('左手手套',l_glove_angle_np.shape)
# print('右手手套',r_glove_angle_np.shape)
total_frames = l_joint_angle_np.shape[0]
env = gym.make('yumi-v0')
observation = env.reset()
camera_distance = 2
camera_yaw = 90
camera_pitch = -10
camera_roll = 0
camera_target_position = [0, 0, 1.0]
paused = False
v_rate = 10

# robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# handle = robot.rm_create_robot_arm("192.168.10.19", 8080)
# print("机械臂ID,", handle.id)
# min = [-178.0, -130.0, -135.0, -180.0, -128.0, -360.0]
# max = [178.0, 130.0, 135.0, 180.0, 128.0, 360.0]
# L_robot_arm_angle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
L_robot_arm_angle = [0.0, 0.0, 1.57, 0.0, 0.0, 0.0]
R_robot_arm_angle = [0.0, 0.0, -1.57, 0.0, 0.0, 0.0]
stop = False
while not stop:
    env.render()
    for t in range(total_frames):
        for i in range(2):
            # temp = l_joint_angle_np[t].copy()
            # temp = temp.tolist()
            # L_robot_arm_angle = trans2realworld(temp,min,max)
            # print(robot.rm_movej(L_robot_arm_angle, 20, 0, 0, 1))
            L_robot_arm_angle = l_joint_angle_np[t].tolist()
            R_robot_arm_angle = r_joint_angle_np[t].tolist()
            # print('左臂',L_robot_arm_angle)
            # print('右臂',R_robot_arm_angle)
            # R_robot_arm_angle[-1] = t/180.0*3.14
            # L_robot_arm_angle[-1] = t/180.0*3.14
            # L_robot_arm_angle[0:5] = [0, 0, 1.57, 0, 0]
            # R_robot_arm_angle[0:5] = [0, 0, -1.57, 0, 0]
            # L_robot_arm_angle = [90, 90, 90, 90, 90, 90]*math.pi/180
            # R_robot_arm_angle = [90, 90, 90, 90, 90, 90]*math.pi/180
            action = L_robot_arm_angle + R_robot_arm_angle 
            # action = L_robot_arm_angle + R_robot_arm_angle + l_glove_angle_np[t].tolist() + r_glove_angle_np[t].tolist()
        #     # action = l_joint_angle_np[t] + r_joint_angle_np[t] + r_glove_angle_np[t] + l_glove_angle_np[t]
            keys = p.getKeyboardEvents()
            for k, v in keys.items():
                if v & p.KEY_WAS_TRIGGERED:
                    if k == ord('w'):
                        camera_distance -= 0.3
                    elif k == ord('s'):
                        camera_distance += 0.3
                    elif k == ord('a'):
                        camera_yaw -= 10
                    elif k == ord('d'):
                        camera_yaw += 10
                    elif k == ord('q'):
                        camera_pitch -= 10
                    elif k == ord('e'):
                        camera_pitch += 10
                    elif k == ord(' '):
                        paused = not paused
                        print ('切换暂停')
                        # print ('当前帧数：',t)
                    # elif k == ord('8'):
                    #     v_rate = v_rate *2
                    #     print ('速度下降',v_rate)
                    # elif k == ord('2'):
                    #     v_rate = v_rate/2
                    #     print ('速度提高',v_rate)
                    # elif k == ord('u'):
                    #     camera_target_position[0] += 0.05  # 向右移动
                    # elif k == ord('j'):
                    #     camera_target_position[0] -= 0.05  # 向左移动
                    # elif k == ord('i'):
                    #     camera_target_position[1] += 0.05  # 向前移动
                    # elif k == ord('k'):
                    #     camera_target_position[1] -= 0.05  # 向后移动
                    # elif k == ord('o'):
                    #     camera_target_position[2] += 0.05  # 向上移动
                    # elif k == ord('l'):
                    #     camera_target_position[2] -= 0.05  # 向下移动
                    # elif k == ord('b'):
                    #     stop = True
                    #     break
        #     # 如果处于暂停状态，则跳过仿真步骤
            if paused:
                time.sleep(0.02)  # 保持短暂延迟以减少CPU占用
        #         conti、nue

            p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                            cameraYaw=camera_yaw,
                                            cameraPitch=camera_pitch,
                                            cameraTargetPosition=camera_target_position)
            observation, reward, done, info = env.step(action)
            time.sleep(0.02*v_rate)
        print('当前帧数：',t)
    # #esc 退出
    # stop = True
    # if keyboard.is_pressed('b'):
    #    stop = True
# RoboticArm.rm_destroy()

