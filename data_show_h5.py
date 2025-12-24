#展示h5文件中人体关节点数据

# show_data_h5.py
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial.transform import Rotation as R


offset = 0.2
# 关节名称对应字典
joint_names = [
    "Pelvis", "Left Hip", "Right Hip", "Spine1", "Left Knee", "Right Knee",
    "Spine2", "Left Ankle", "Right Ankle", "Spine3", "Left Foot", "Right Foot",
    "Neck", "Left Collar", "Right Collar", "Head", "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hand", "Right Hand"
]
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

def draw_coordinate_axes(ax, position, quaternion, scale=0.1):
    """
    在给定位置绘制坐标轴
    :param ax: 3D绘图轴
    :param position: 关节位置 [x, y, z]
    :param quaternion: 四元数 [qx, qy, qz, qw]
    :param scale: 坐标轴长度缩放因子
    """
    # 创建旋转对象
    rotation = R.from_quat(quaternion)
    
    # 定义初始坐标轴 (单位向量)
    axes = np.array([
        [1, 0, 0],  # X轴 - 红色
        [0, 1, 0],  # Y轴 - 绿色
        [0, 0, 1]   # Z轴 - 蓝色
    ])
    
    # 应用旋转
    rotated_axes = rotation.apply(axes)
    
    # 缩放坐标轴
    rotated_axes *= scale
    
    # 绘制坐标轴
    colors = ['r', 'g', 'b']
    for i, (axis, color) in enumerate(zip(rotated_axes, colors)):
        ax.plot([position[0], position[0] + axis[0]], 
                [position[1], position[1] + axis[1]], 
                [position[2], position[2] + axis[2]], 
                color=color, linewidth=2)

def load_and_visualize_h5_data(filename='data.h5'):
    """
    逐帧展示h5文件中的人体关节点数据
    """
    # 打开h5文件
    with h5py.File(filename, 'r') as h5f:
        # 获取时间戳数据
        timestamps = h5f['timestamp'][:]
        num_frames = len(timestamps)
        
        print(f"加载到 {num_frames} 帧数据")
        
        # 创建3D图形窗口
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置坐标轴范围和标签
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-3+offset, 1+offset)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Body Tracking Data from H5 File')
        
        # 逐帧显示数据
        for frame_idx in range(num_frames):
            # 清除之前的点
            ax.cla()
            
            # 重新设置坐标轴范围和标签
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-2+offset, 0+offset)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Body Tracking Data - Frame {frame_idx+1}/{num_frames} - Time: {timestamps[frame_idx]:.0f} ns')
            
            # 收集当前帧所有关节的位置数据
            positions = []
            quaternions = []
            for name in joint_names:
                # 获取关节数据 (x, y, z, qx, qy, qz, qw)
                joint_data = h5f[name][frame_idx]
                # 取位置信息 (x, y, z)
                positions.append(joint_data[:3])
                # 取四元数信息 (qx, qy, qz, qw)
                quaternions.append(joint_data[3:])
            
            # 转换为numpy数组
            positions = np.array(positions)
            positions = trans_xyz(positions)
            quaternions = np.array(quaternions)

            # 对四元数进行坐标系转换
            transformed_quaternions = []
            for quat in quaternions:
                transformed_quat = trans_quat(quat)
                transformed_quaternions.append(transformed_quat)
            transformed_quaternions = np.array(transformed_quaternions)
            
            # 绘制所有关节
            xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
            ax.scatter(xs, ys, zs, c='r', marker='o', s=50)
            
            # 为每个关节点添加标签
            for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
                ax.text(x, y, z, f'{joint_names[i]}', fontsize=6)
            
            # 绘制骨骼连线（可选）
            draw_skeleton(ax, positions)
            
            # 绘制每个关节的坐标轴
            for pos, quat in zip(positions, transformed_quaternions):
                draw_coordinate_axes(ax, pos, quat, scale=0.1)
            # 刷新图形
            plt.draw()
            plt.pause(0.5)  # 控制播放速度，可根据需要调整
            
            # 检查是否关闭了图形窗口
            if not plt.fignum_exists(fig.number):
                break
                
        print("播放完成")

def draw_skeleton(ax, positions):
    """
    绘制人体骨骼连线
    """
    # 定义骨骼连接关系
    skeleton_connections = [
        # 脊柱和头部
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # 脊柱连接
        # 左臂
        (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),
        # 右臂
        (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),
        # 左腿
        (0, 1), (1, 4), (4, 7), (7, 10),
        # 右腿
        (0, 2), (2, 5), (5, 8), (8, 11)
    ]
    
    # 绘制连线
    for start_idx, end_idx in skeleton_connections:
        if start_idx < len(positions) and end_idx < len(positions):
            x_coords = [positions[start_idx, 0], positions[end_idx, 0]]
            y_coords = [positions[start_idx, 1], positions[end_idx, 1]]
            z_coords = [positions[start_idx, 2], positions[end_idx, 2]]
            ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=1, alpha=0.7)

if __name__ == "__main__":
    try:
        load_and_visualize_h5_data('test_1000.h5')
    except FileNotFoundError:
        print("未找到 data.h5 文件，请确保文件存在")
    except Exception as e:
        print(f"发生错误: {e}")