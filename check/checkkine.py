import torch
import torch.nn as nn
from torch_geometric.data import Data
from urdfpy import URDF, matrix_to_xyz_rpy
import math
from utils.urdf2graph import yumi2graph, hand2graph
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from models.kinematics import ForwardKinematicsURDF,ForwardKinematicsAxis

def update_angle(x, joint_idx, angle):
    """
    更新指定关节的角度
    :param x: 输入角度张量
    :param joint_idx: 要更新的关节索引
    :param angle: 新的角度值 (弧度)
    :return: 更新后的张量
    """
    x[0][joint_idx] = angle
    return x

def draw_coordinate_axes(ax, origin, rot,rot_axis, scale=0.1):
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
    rot_axis = rot@ rot_axis
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
    
    # 绘制旋转轴(黑色)
    ax.plot([origin[0], origin[0] + rot_axis[0] * scale*1.3],
            [origin[1], origin[1] + rot_axis[1] * scale*1.3],
            [origin[2], origin[2] + rot_axis[2] * scale*1.3], 'k')

def animate(frame, x, graph, fk, ax, edge_index, joint_texts):
    ax.cla()  # 清空当前坐标轴

    # 设置视角和坐标轴范围
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-0.2, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)
    
    frame_sampleRate=50
    # 更新角度（遍历每个关节）
    num_joints = x.shape[1]
    joint_idx = frame // frame_sampleRate % num_joints  # 每个关节停留10帧
    angle = (frame % frame_sampleRate) * (2 * np.pi / frame_sampleRate)  # 每次增加 36°
    if (frame-1) // frame_sampleRate % num_joints != joint_idx:
        temp =(frame-1) // frame_sampleRate % num_joints
        angle = 0  #角度归0
        x = update_angle(x, temp, angle)
    x = update_angle(x, joint_idx, angle)
    pos1, rot, pos = fk(x, graph.parent, graph.offset, 1,graph.axis)
    rot_axis=graph.axis
    
    # 绘制关节连接线
    for edge in edge_index.permute(1, 0):
        line_x = [pos[edge[0]][0], pos[edge[1]][0]]
        line_y = [pos[edge[0]][1], pos[edge[1]][1]]
        line_z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(line_x, line_y, line_z, 'royalblue', marker='o')
    
    # 绘制每个关节的坐标轴
    for i, joint in enumerate(pos):
        origin = joint.tolist()
        draw_coordinate_axes(ax, origin, rot[i],rot_axis[i], scale=0.05)  # scale 根据需要调整

    # 更新关节索引标注
    for i, joint in enumerate(pos):
        x_j, y_j, z_j = joint.tolist()
        joint_texts[i].set_position((x_j, y_j))
        joint_texts[i].set_3d_properties(z_j)
    

    ax.set_title(f"Joint {joint_idx} Angle: {angle:.2f} rad")



if __name__ == '__main__':
    yumi_cfg = {
        'joints_name': [
            'left_shoulder_pitch_joint',
            'left_shoulder_roll_joint',
            'left_shoulder_yaw_joint',
            'left_elbow_joint',
            'left_hand_joint',
            'L_base_link_joint',

            'right_shoulder_pitch_joint',
            'right_shoulder_roll_joint',
            'right_shoulder_yaw_joint',
            'right_elbow_joint',
            'right_hand_joint',
            'R_base_link_joint',
        ],
        'edges': [
            ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint'],
            ['left_shoulder_roll_joint', 'left_shoulder_yaw_joint'],
            ['left_shoulder_yaw_joint', 'left_elbow_joint'],
            ['left_elbow_joint', 'left_hand_joint'],
            ['left_hand_joint', 'L_base_link_joint'],


            ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint'],
            ['right_shoulder_roll_joint', 'right_shoulder_yaw_joint'],
            ['right_shoulder_yaw_joint', 'right_elbow_joint'],
            ['right_elbow_joint', 'right_hand_joint'],
            ['right_hand_joint', 'R_base_link_joint'],

        ],
        'root_name': [
            'left_shoulder_pitch_joint',
            'right_shoulder_pitch_joint',
        ],
        'end_effectors': [
            'left_hand_joint',
            'right_hand_joint',
        ],
        'shoulders': [
            'left_shoulder_roll_joint',
            'right_shoulder_roll_joint',
        ],
        'elbows': [
            'left_elbow_joint',
            'right_elbow_joint',
        ],
    }

    graph = yumi2graph(urdf_file="D:\\2025\\crp\\new_robot- YUSHU_dist\\data\\target\\yumi-all\h1_with_hand.urdf", cfg=yumi_cfg)
    
    fk = ForwardKinematicsAxis()
    # 初始化角度输入
    x = torch.zeros(1, len(yumi_cfg['joints_name']))
    pos1,rot,pos= fk(x, graph.parent, graph.offset, 1,graph.axis)

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 初始化关节标注
    edge_index = graph.edge_index
    joint_texts = [ax.text(0, 0, 0, f'{i}', fontsize=10, color='black') for i in range(len(graph.x))]

    # 创建动画
    ani = FuncAnimation(fig, animate, fargs=(x, graph, fk, ax, edge_index, joint_texts),
                        interval=400,  # 每帧间隔时间（毫秒）
                        blit=False, cache_frame_data=False)
    # matplotlib.rcParams['animation.ffmpeg_path'] = "D:\software\\ffmpeg-7.1.1-full_build\\ffmpeg-7.1.1-full_build\\bin\\ffmpeg.exe"
    # ani.save('animation.mp4', writer='ffmpeg', fps=30, dpi=150)
    # 控制动画播放的变量
    pause = False

    def on_press(event):
        global pause
        if event.key == ' ':
            pause = not pause
            if pause:
                ani.event_source.stop()
            else:
                ani.event_source.start()
    # 绑定键盘事件
    fig.canvas.mpl_connect('key_press_event', on_press)
    plt.show()




