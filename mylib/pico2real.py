# import h5py
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
# 身高偏置
offset = 0.2
def trans_xyz(position):
    '''
    从头显坐标系转为大地坐标系
    新x=z,新y=x,新z=y
    '''
    temp = position.copy()
    temp[:, 1] = -position[:, 0]
    temp[:, 2] = position[:, 1]+offset
    temp[:, 0] = -position[:, 2]
    return temp

def trans_quat(quaternion):
    '''
    将四元数从头显坐标系转换到大地坐标系
    通过欧拉角进行坐标变换
    坐标变换规则: 新x=z, 新y=x, 新z=y
    '''
    # 创建旋转对象
    rot = R.from_quat(quaternion)
    
    # 转换为欧拉角 (XYZ顺序)
    euler = rot.as_euler('xyz')
    
    # 应用坐标变换: 新x=z, 新y=x, 新z=y
    # 这意味着我们将原z轴的旋转变为新x轴的旋转
    # 原x轴的旋转变为新y轴的旋转
    # 原y轴的旋转变为新z轴的旋转
    new_euler = np.array([-euler[2], -euler[0], euler[1]])
    
    # 转换回四元数
    new_rot = R.from_euler('xyz', new_euler)
    return new_rot.as_quat()

def get_palm_vector(quaternion):
    """
    获取手掌掌心向量（X轴向量）
    :param quaternion: 四元数 [qx, qy, qz, qw]
    :return: 掌心向量 [x, y, z]
    """
    # 创建旋转对象
    rotation = R.from_quat(quaternion)
    
    # 定义初始X轴向量
    x_axis = np.array([0, 0, -1])
    
    # 应用旋转得到掌心向量
    palm_vector = rotation.apply(x_axis)
    
    return palm_vector

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
    #保留五位小数
    angle_real = angle_real.round(5)
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
    
    #转为list
    #保留五位小数
    angle_real = angle_real.round(5)
    angle_real = angle_real.tolist()
    print(angle_real)
    return angle_real
def trans2realworld_hand(angle,min=0,max=0):
    '''
    要转换为角度,且检查是否超限,输入为弧度,下限,上限
    '''
    angle_real = (angle+3)/4*2000
    return angle_real
