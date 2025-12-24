# 数据位于data/source/目录下
# 该脚本用于检查数据的完整性和格式
import torch
import torch.nn as nn
from torch_geometric.data import Data
from urdfpy import URDF, matrix_to_xyz_rpy

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

def get_frame_data(data_group, yumi_cfg, frame):
    pos_dict = {}
    rot_dict = {}
    for joint in yumi_cfg['joints_pos']:
        if joint not in data_group:
            raise ValueError(f"Joint '{joint}' not found in HDF5 file")
        pos_dict[joint] = data_group[joint][frame]
    for joint in yumi_cfg['joints_rot']:
        if joint not in data_group:
            raise ValueError(f"Joint '{joint}' not found in HDF5 file")
        rot_dict[joint] = data_group[joint][frame]
    l_hand_pos, r_hand_pos = data_group['l_glove_pos'][frame], data_group['r_glove_pos'][frame]
    #给左右手的位置最前面加一个零点
    l_hand_pos = np.concatenate((np.zeros((1,3)),l_hand_pos))
    r_hand_pos = np.concatenate((np.zeros((1,3)),r_hand_pos))
    # 反转 左手的y

    return pos_dict, rot_dict, l_hand_pos, r_hand_pos

def draw_coordinate_axes(ax, origin, rot, scale=0.1,rot_axis=[0,0,1],):
    """
    绘制正交的 XYZ 坐标轴
    :param ax: matplotlib 的 3D 坐标轴对象
    :param origin: 坐标轴原点 [x, y, z]
    :param rot: 旋转矩阵 [3, 3]
    :param scale: 坐标轴长度缩放因子
    """
    x_axis = rot[:, 0]  # X 轴方向
    y_axis = rot[:, 1]  # Y 轴方向
    z_axis = rot[:, 2]  # Z 轴方向
    # 计算旋转矩阵处理后的旋转轴,将旋转矩阵乘以旋转轴
    # rot_axis = rot@ rot_axis
    # 绘制 X 轴（红色）
    ax.plot([origin[0], origin[0] + x_axis[0] * scale],
            [origin[1], origin[1] + x_axis[1] * scale],
            [origin[2], origin[2] + x_axis[2] * scale], 'r')

    # 绘制 Y 轴（绿色）
    ax.plot([origin[0], origin[0] + y_axis[0] * scale],
            [origin[1], origin[1] + y_axis[1] * scale],
            [origin[2], origin[2] + y_axis[2] * scale], 'g')

    # 绘制 Z 轴（蓝色）
    ax.plot([origin[0], origin[0] + z_axis[0] * scale],
            [origin[1], origin[1] + z_axis[1] * scale],
            [origin[2], origin[2] + z_axis[2] * scale], 'b')
    
    # # 绘制旋转轴(黑色)
    # ax.plot([origin[0], origin[0] + rot_axis[0] * scale*1.3],
    #         [origin[1], origin[1] + rot_axis[1] * scale*1.3],
    #         [origin[2], origin[2] + rot_axis[2] * scale*1.3], 'k')
def animate(frame):
    global data_group, ax, data_cfg

    # 清除当前轴
    ax.cla()

    # 设置3D视图
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.4, 0.4])
    ax.set_zlim([-0.6,0.1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title(f'3D Animation Frame: {frame}')

    # 创建位置字典，从各个关节数据集中读取当前帧的位置
    # pos_dict = {}
    # for joint in yumi_cfg['joints_pos']:
    #     # 检查数据集是否存在
    #     if joint not in data_group:
    #         raise ValueError(f"Joint '{joint}' not found in HDF5 file")
    #     # 读取当前帧数据
    #     joint_data = data_group[joint][frame]
    #     pos_dict[joint] = joint_data
    pos_dict , rot_dict,l_hand_pos,r_hand_pos = get_frame_data(data_group, data_cfg, frame)
    # 绘制骨架
    for edge in data_cfg['edges']:
        start_joint, end_joint = edge
        start_pos = pos_dict[start_joint]
        end_pos = pos_dict[end_joint]

        # 绘制骨骼线
        ax.plot([start_pos[0], end_pos[0]], 
                [start_pos[1], end_pos[1]], 
                [start_pos[2], end_pos[2]], 'r-')
    
    #绘制每个关节的坐标轴

    for joint_name in data_cfg['joints_name']:
        #拼接关节名称和pos和rot
        joint_pos_key = f"{joint_name}_pos"
        origin = pos_dict[joint_pos_key].tolist()
        joint_rot_key = f"{joint_name}_quat"
        joint_rot = rot_dict[joint_rot_key]
        #将四元数转换为旋转矩阵
        rot_matrix = quat2rot(joint_rot)
        if joint_name == 'l_hd':
            rotated_points = rot_matrix @ l_hand_pos.T
            origin_array = np.array(origin)
            global_hand_pos = rotated_points.T + origin_array
            l_hand_pos = global_hand_pos
        if joint_name == 'r_hd':
            rotated_points = rot_matrix @ r_hand_pos.T
            origin_array = np.array(origin)
            global_hand_pos = rotated_points.T + origin_array
            r_hand_pos = global_hand_pos
        draw_coordinate_axes(ax, origin, rot_matrix, scale=0.05)  # scale 根据需要调整
    # 绘制关节点
    for joint, pos in pos_dict.items():
        ax.scatter(pos[0], pos[1], pos[2], c='b', marker='o')
    #将手掌的位置应用旋转矩阵

    # 绘制手部
    ax.scatter(l_hand_pos[:,0], l_hand_pos[:,1], l_hand_pos[:,2], c='g', marker='o',s=5)
    # 根据cfg绘制连接
    for edge in data_cfg['hand_edges']:
        start_joint, end_joint = edge
        start_pos = l_hand_pos[start_joint]
        end_pos = l_hand_pos[end_joint]
        # 绘制骨骼线
        ax.plot([start_pos[0], end_pos[0]], 
                [start_pos[1], end_pos[1]], 
                [start_pos[2], end_pos[2]], 'g-')
        
    ax.scatter(r_hand_pos[:,0], r_hand_pos[:,1], r_hand_pos[:,2], c='r', marker='o',s=5)
    for edge in data_cfg['hand_edges']:
        start_joint, end_joint = edge
        start_pos = r_hand_pos[start_joint]
        end_pos = r_hand_pos[end_joint]
        # 绘制骨骼线
        ax.plot([start_pos[0], end_pos[0]], 
                [start_pos[1], end_pos[1]], 
                [start_pos[2], end_pos[2]], 'r')

    # 添加关节名称标签
    for joint, pos in pos_dict.items():
        ax.text(pos[0], pos[1], pos[2], joint, fontsize=10)

def quat2rot(quat):
    """
    将四元数转换为旋转矩阵
    :param quat: 四元数 [x, y, z, w]
    :return: 旋转矩阵 [3, 3]
    """
    x, y, z, w = quat
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])
#暂停处理
def on_press(event):
    global pause
    if event.key == ' ':
        pause = not pause
        if pause:
            ani.event_source.stop()
        else:
            ani.event_source.start()

if __name__ == '__main__':
    #先读取h5数据
    
    # h5_file = h5py.File('D:\\2025\\crp\\new_robot- YUSHU_dist\data\source\sign-all\\train\h5\\banliyewu_YuMi.h5', 'r')
    h5_file = h5py.File('D:\\2025\\crp\\ur3_robot_train\\data\\source\\sign-all\\train\\h5\\banliyewu_YuMi.h5', 'r')
    data_cfg = {
        'joints_name': [
            'l_fr',
            'l_hd',
            'l_up',

            'r_fr',
            'r_hd',
            'r_up',
        ],
        'edges': [
            ['l_fr_pos', 'l_hd_pos'],
            ['l_up_pos', 'l_fr_pos'],


            ['r_fr_pos', 'r_hd_pos'],
            ['r_up_pos', 'r_fr_pos'],
        ],
        'joints_pos': [
            'l_fr_pos',
            'l_hd_pos',
            'l_up_pos',

            'r_fr_pos',
            'r_hd_pos',
            'r_up_pos',
        ],
        'joints_rot':[
            'l_fr_quat',
            'l_hd_quat',
            'l_up_quat',

            'r_fr_quat',
            'r_hd_quat',
            'r_up_quat',
        ],
        'hand_edges': [
            (0, 1), (1, 2), (2, 3),           # 第一根手指
            (0, 4), (4, 5), (5, 6),           # 第二根手指
            (0, 7), (7, 8), (8, 9),           # 第三根手指
            (0, 10), (10, 11), (11, 12),      # 第四根手指
            (0, 13), (13, 14), (14, 15), (15, 16)  # 第五根手指（拇指）
        ]
    }
    key ='再见-zaijian'
    # key ='时尚-shishang'
    data_group=  h5_file[key]
    data_time = data_group['time']
    #获取帧数量
    frame_num = data_time.shape[0]
    
    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #动画对象实例
    ##补充区域开始
        # 创建动画
    ani = FuncAnimation(fig, animate, frames=frame_num, interval=250, repeat=True)
    ##补充区域截止

    pause = False
    
    # 绑定键盘事件
    fig.canvas.mpl_connect('key_press_event', on_press)
    # 显示图形
    plt.show()
