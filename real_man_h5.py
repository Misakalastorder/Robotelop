import numpy as np
import h5py
import time
from Robotic_Arm.rm_robot_interface import *
import math
import keyboard
import gym, yumi_gym
import pybullet as p
def trans2realworld(angle,min,max):
    '''
    要转换为角度,且检查是否超限,输入为弧度,下限,上限
    '''
    #检查长度是否一致
    # 将 angle 转换为 numpy 数组以支持数值运算
    if len(angle) != len(min) or len(angle) != len(max):
        assert('长度不一致')
        return 0
    angle_np = np.array(angle)
    angle_real = angle_np * 180.0 / math.pi
    angle_real[5] = -angle_real[5]
    # 使用 enumerate 来获取索引和值
    # 将首个角度取反，因为第一个角度为负
    # angle_real[0] = (180.0+angle_real[0])%180.0
    # angle_real= - angle_real
    # if angle_real[0] < 0:
    #     angle_real[0] = (180.0+angle_real[0])
    # else:
    #     angle_real[0] = (-180.0+angle_real[0])
    for i, ang in enumerate(angle_real):
        if ang > max[i]:
            angle_real[i] = max[i] - 1
            print('角度过大')
        elif ang < min[i]:
            angle_real[i] = min[i] + 1
            print('角度过小')
    
    #转为list
    angle_real = np.round(angle_real, 2)
    angle_real = angle_real.tolist()
    print(angle_real)
    return angle_real

def trans2realworld1(angle,min,max):
    '''
    要转换为角度,且检查是否超限,输入为弧度,下限,上限
    '''
    #检查长度是否一致
    # 将 angle 转换为 numpy 数组以支持数值运算
    if len(angle) != len(min) or len(angle) != len(max):
        assert('长度不一致')
        return 0
    angle_np = np.array(angle)
    angle_real = angle_np * 180.0 / math.pi
    
    # angle_real= - angle_real
    angle_real[1] = -angle_real[1]
    angle_real[2] = -angle_real[2]
    angle_real[4] = -angle_real[4]
    angle_real[5] = -angle_real[5]
    for i, ang in enumerate(angle_real):
        if ang > max[i]:
            angle_real[i] = max[i] - 1
            print('角度过大')
        elif ang < min[i]:
            angle_real[i] = min[i] + 1
            print('角度过小')
    # 保留四位小数 防止溢出
    angle_real = np.round(angle_real, 2)
    #转为list
    angle_real = angle_real.tolist()
    print(angle_real)
    return angle_real
def trans2realworld_hand(angle,min=0,max=0):
    '''
    要转换为角度,且检查是否超限,输入为弧度,下限,上限
    '''
    angle_real = (angle+3)/4*2000
    return angle_real

hf = h5py.File("D:\\2025\\crp\\ur3_robot_train_pico\\saved\\h5\\测试-ceshi_1111.h5", 'r') #朝向
group = hf.get('group1')
l_joint_angle = group.get('l_joint_angle')
r_joint_angle = group.get('r_joint_angle')
l_hand_angle = group.get('l_glove_angle')
r_hand_angle = group.get('r_glove_angle')
print('左手动作',l_joint_angle)
l_joint_angle_np = np.array(l_joint_angle)
r_joint_angle_np = np.array(r_joint_angle)
print('左手手臂',l_joint_angle_np.shape)
print('右手手臂',r_joint_angle_np.shape)

total_frames = l_joint_angle_np.shape[0]

l_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = l_arm.rm_create_robot_arm("192.168.10.19", 8080)
r_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
r_handle = r_arm.rm_create_robot_arm("192.168.10.18", 8080)
# print("机械臂ID,", handle.id)
min = [-178.0, -130.0, -135.0, -180.0, -128.0, -360.0]
max = [178.0, 130.0, 135.0, 180.0, 128.0, 360.0]
# L_robot_arm_angle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# R_robot_arm_angle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
L_robot_arm_angle = [0.0, -90, 0, 0.0, 0.0, 0.0]
R_robot_arm_angle = [0.0, -90, 0, 0.0, 0.0, 0.0]
l_arm.rm_movej(L_robot_arm_angle, 10, 0, 0, 1)
r_arm.rm_movej(R_robot_arm_angle, 10, 0, 0, 1)
stop = False
# l_arm.rm_movej([0, 0, 0, 0, 0, 0], 50, 0, 0, 1)
# r_arm.rm_movej([0, 0, 0, 0, 0, 0], 50, 0, 0, 1)
# L_robot_hand_angle = [950,950,950,500,600,500]
# L_robot_hand_angle = [900,900,900,900,900,700]
# R_robot_hand_angle = [200,200,200,200,200,500]
select_frame = 585
while not stop:
    l_arm.rm_movej(trans2realworld1(l_joint_angle_np[0],min,max), 50, 0, 0, 1)
    r_arm.rm_movej(trans2realworld(r_joint_angle_np[0],min,max), 50, 0, 0, 1)
    # l_arm.rm_movej(trans2realworld1(l_joint_angle_np[select_frame],min,max), 50, 0, 0, 1)
    # r_arm.rm_movej(trans2realworld(r_joint_angle_np[select_frame],min,max), 50, 0, 0, 1)
    for i in range(total_frames):
        print(i) 
        L_robot_arm_angle = trans2realworld1(l_joint_angle_np[i], min, max)
        R_robot_arm_angle = trans2realworld(r_joint_angle_np[i], min, max)
        # L_robot_arm_angle = trans2realworld1(l_joint_angle_np[select_frame], min, max)
        # R_robot_arm_angle = trans2realworld(r_joint_angle_np[select_frame], min, max)
        # R_robot_hand_angle = trans2realworld_hand(r_hand_angle[i])
        # L_robot_arm_angle = [90, 90, 90, 90, 90, 90]
        # R_robot_arm_angle = [0, 0, 0, 0, 0, 0]
        # L_robot_arm_angle[0:5] = [0, 0, -90, 0, 0]
        # R_robot_arm_angle[0:5] = [0, 0, -90, 0, 0]
        # R_robot_arm_angle[-1] = i
        # L_robot_arm_angle[-1] = i

        print('左臂命令',l_arm.rm_movej_canfd(L_robot_arm_angle, False, 0, 1, 50),'原始角度',L_robot_arm_angle)
        time.sleep(0.05)
        # print('左手命令',l_arm.rm_set_hand_angle(L_robot_hand_angle,True,1))
        # time.sleep(0.05)
        # print('右手命令',r_arm.rm_set_hand_angle(L_robot_hand_angle,True,1))
        print('右臂命令',r_arm.rm_movej_canfd(R_robot_arm_angle, False, 0, 1, 50),'原始角度',R_robot_arm_angle)
        time.sleep(0.05)
        # L_robot_hand_angle = [1000,1000,1000,1000,1000,500]
        
        # print(r_arm.rm_set_hand_angle(L_robot_hand_angle,True,1))
        # print(L_robot_hand_angle)
        # print('右手命令',r_arm.rm_set_hand_follow_angle([0,100,200,300,400,500], True))
        # print('右手命令',r_arm.rm_set_hand_follow_angle(R_robot_hand_angle, True))
#     # 获取当前关节角度
        # print('当前',l_arm.rm_get_joint_degree())

        #等待键盘输入 阻塞等待
        # key_value = keyboard.read_key()
        # if key_value == 'b':
        #     stop = True
        #     break
        # elif key_value == 'n':
        #     continue
        # elif key_value == 'q':
        #     L_robot_hand_angle[0] = L_robot_hand_angle[0] + 50
        # elif key_value == 'a':
        #     L_robot_hand_angle[0] = L_robot_hand_angle[0] - 50
        # elif key_value == 'w':
        #     L_robot_hand_angle[1] = L_robot_hand_angle[1] + 50
        # elif key_value == 's':
        #     L_robot_hand_angle[1] = L_robot_hand_angle[1] - 50
        # elif key_value == 'e':
        #     L_robot_hand_angle[2] = L_robot_hand_angle[2] + 50
        # elif key_value == 'd':
        #     L_robot_hand_angle[2] = L_robot_hand_angle[2] - 50
        # elif key_value == 'r':
        #     L_robot_hand_angle[3] = L_robot_hand_angle[3] + 50
        # elif key_value == 'f':
        #     L_robot_hand_angle[3] = L_robot_hand_angle[3] - 50
        # elif key_value == 't':
        #     L_robot_hand_angle[4] = L_robot_hand_angle[4] + 50
        # elif key_value == 'g':
        #     L_robot_hand_angle[4] = L_robot_hand_angle[4] - 50
        # elif key_value == 'y':
        #     L_robot_hand_angle[5] = L_robot_hand_angle[5] + 50
        # elif key_value == 'h':
        #     L_robot_hand_angle[5] = L_robot_hand_angle[5] - 50
        
        if stop: break
    stop = True
# 断开所有连接
# l_arm.rm_movej([0.0, -90, 0, 0.0, 0.0, 0.0], 20, 0, 0, 1)
# # l_arm.rm_set_hand_follow_angle([0,0,0,0,0,0], True)
# r_arm.rm_movej([0.0, -90, 0, 0.0, 0.0, 0.0], 20, 0, 0, 1)
RoboticArm.rm_destroy()