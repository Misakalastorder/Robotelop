import torch
import torch.nn as nn
from torch_geometric.data import Data
# from urdfpy import URDF, matrix_to_xyz_rpy
from urchin import URDF, matrix_to_xyz_rpy
import math

import numpy as np


"""
Forward Kinematics with Different Axes
"""
class ForwardKinematicsAxis(nn.Module):
    def __init__(self):
        super(ForwardKinematicsAxis, self).__init__()

    def forward(self, x, parent, offset, num_graphs, axis, order='xyz'):
        """
        x -- joint angles [num_graphs*num_nodes, 1]
        parent -- node parent [num_graphs*num_nodes]
        offset -- node origin(xyzrpy) [num_graphs*num_nodes, 6]
        num_graphs -- number of graphs
        axis -- rotation axis for rotation x
        order -- rotation order for init rotation
        """
        x = x.view(num_graphs, -1) # [batch_size, num_nodes]
        parent = parent.view(num_graphs, -1)[0] # [num_nodes] the same batch, the same topology
        axis = axis.view(num_graphs, -1, 3)[0] # [num_nodes, 3] the same batch, the same topology
        axis_norm = torch.norm(axis, dim=-1)
        # print(x.shape, axis.shape)
        x = x * axis_norm # filter no rotation node
        offset = offset.view(num_graphs, -1, 6) # [batch_size, num_nodes, 6]
        xyz = offset[:, :, :3] # [batch_size, num_nodes, 3]
        rpy = offset[:, :, 3:] # [batch_size, num_nodes, 3]

        positions = torch.empty(x.shape[0], x.shape[1], 3, device=x.device) # [batch_size, num_nodes, 3]
        global_positions = torch.empty(x.shape[0], x.shape[1], 3, device=x.device) # [batch_size, num_nodes, 3]
        rot_matrices = torch.empty(x.shape[0], x.shape[1], 3, 3, device=x.device) # [batch_size, num_nodes, 3, 3]

        transform = self.transform_from_multiple_axis(x, axis) # [batch_size, num_nodes, 3, 3]
        # print("x11",x)
        # print("transform11",transform)
        # print("axis11",axis)
        # print("offset11",offset)
        rpy_transform = self.transform_from_euler(rpy, order) # [batch_size, num_nodes, 3, 3]
        # print("rpy11",rpy)

        # iterate all nodes
        for node_idx in range(x.shape[1]):
            # serach parent
            parent_idx = parent[node_idx]

            # position
            if parent_idx != -1:
                positions[:, node_idx, :] = torch.bmm(rot_matrices[:, parent_idx, :, :], xyz[:, node_idx, :].unsqueeze(2)).squeeze() + positions[:, parent_idx, :]
                global_positions[:, node_idx, :] = torch.bmm(rot_matrices[:, parent_idx, :, :], xyz[:, node_idx, :].unsqueeze(2)).squeeze() + global_positions[:, parent_idx, :]
                rot_matrices[:, node_idx, :, :] = torch.bmm(rot_matrices[:, parent_idx, :, :].clone(), torch.bmm(rpy_transform[:, node_idx, :, :], transform[:, node_idx, :, :]))
            else:
                positions[:, node_idx, :] = torch.zeros(3) # xyz[:, node_idx, :]
                global_positions[:, node_idx, :] = xyz[:, node_idx, :]
                rot_matrices[:, node_idx, :, :] = torch.bmm(rpy_transform[:, node_idx, :, :], transform[:, node_idx, :, :])

        return positions.view(-1, 3), rot_matrices.view(-1, 3, 3), global_positions.view(-1, 3)

    @staticmethod
    def transform_from_euler(rotation, order):
        transform = torch.matmul(ForwardKinematicsAxis.transform_from_single_axis(rotation[..., 2], order[2]),
                                 ForwardKinematicsAxis.transform_from_single_axis(rotation[..., 1], order[1]))
        transform = torch.matmul(transform,
                                 ForwardKinematicsAxis.transform_from_single_axis(rotation[..., 0], order[0]))
        return transform

    @staticmethod
    def transform_from_single_axis(euler, axis):
        transform = torch.empty(euler.shape[0:2] + (3, 3), device=euler.device) # [batch_size, num_nodes, 3, 3]
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform

    @staticmethod
    def transform_from_multiple_axis(euler, axis):
        transform = torch.empty(euler.shape[0:2] + (3, 3), device=euler.device) # [batch_size, num_nodes, 3, 3]
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        n1 = axis[..., 0]
        n2 = axis[..., 1]
        n3 = axis[..., 2]

        transform[..., 0, 0] = cos + n1 * n1 * (1 - cos)
        transform[..., 0, 1] = n1 * n2 * (1 - cos) - n3 * sin
        transform[..., 0, 2] = n1 * n3 * (1 - cos) + n2 * sin
        transform[..., 1, 0] = n1 * n2 * (1 - cos) + n3 * sin
        transform[..., 1, 1] = cos + n2 * n2 * (1 - cos)
        transform[..., 1, 2] = n2 * n3 * (1 - cos) - n1 * sin
        transform[..., 2, 0] = n1 * n3 * (1 - cos) - n2 * sin
        transform[..., 2, 1] = n2 * n3 * (1 - cos) + n1 * sin
        transform[..., 2, 2] = cos + n3 * n3 * (1 - cos)

        return transform
"""
convert Yumi URDF to graph
"""
# 添加绘制坐标轴的函数
def draw_coordinate_axes(ax, position, rotation_matrix, scale=0.05):
    """
    在给定位置绘制坐标轴
    :param ax: 3D绘图轴
    :param position: 关节位置 [x, y, z]
    :param rotation_matrix: 旋转矩阵 3x3
    :param scale: 坐标轴长度缩放因子
    """
    # 定义初始坐标轴 (单位向量)
    axes = np.array([
        [1, 0, 0],  # X轴 - 红色
        [0, 1, 0],  # Y轴 - 绿色
        [0, 0, 1]   # Z轴 - 蓝色
    ])
    
    # 将旋转矩阵转换为numpy数组
    rot_matrix = rotation_matrix.cpu().numpy()
    
    # 应用旋转
    rotated_axes = rot_matrix.dot(axes.T).T
    
    # 缩放坐标轴
    rotated_axes *= scale
    
    # 绘制坐标轴
    colors = ['r', 'g', 'b']
    for i, (axis, color) in enumerate(zip(rotated_axes, colors)):
        ax.plot([position[0], position[0] + axis[0]], 
                [position[1], position[1] + axis[1]], 
                [position[2], position[2] + axis[2]], 
                color=color, linewidth=2)
def yumi2graph(urdf_file, cfg, angle_x = torch.tensor([[0.0 ,1.57 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,-1.57 ,0.0 ,0.0,0.0,0.0]]),figure_window=None):
    # load URDF
    robot = URDF.load(urdf_file)

    # parse joint params
    joints = {}
    for joint in robot.joints:
        # joint atributes
        joints[joint.name] = {'type': joint.joint_type, 'axis': joint.axis,
                              'parent': joint.parent, 'child': joint.child,
                              'origin': matrix_to_xyz_rpy(joint.origin),
                              'lower': joint.limit.lower if joint.limit else 0,
                              'upper': joint.limit.upper if joint.limit else 0}

    # debug msg
    # for name, attr in joints.items():
    #     print(name, attr)

    # skeleton type & topology type
    skeleton_type = 0
    topology_type = 0

    # collect edge index & edge feature
    joints_name = cfg['joints_name']
    joints_index = {name: i for i, name in enumerate(joints_name)}
    edge_index = []
    edge_attr = []
    for edge in cfg['edges']:
        parent, child = edge
        # add edge index
        edge_index.append(torch.LongTensor([joints_index[parent], joints_index[child]]))
        # add edge attr
        edge_attr.append(torch.Tensor(joints[child]['origin']))
    edge_index = torch.stack(edge_index, dim=0)
    edge_index = edge_index.permute(1, 0)
    edge_attr = torch.stack(edge_attr, dim=0)
    # print(edge_index, edge_attr, edge_index.shape, edge_attr.shape)

    # number of nodes
    num_nodes = len(joints_name)
    # print(num_nodes)

    # end effector mask
    ee_mask = torch.zeros(len(joints_name), 1).bool()
    for ee in cfg['end_effectors']:
        ee_mask[joints_index[ee]] = True

    
    # # pree mask
    # pree_mask = torch.zeros(len(joints_name), 1).bool()
    # for pree in cfg['pree']:
    #     pree_mask[joints_index[pree]] = True

    # shoulder mask
    sh_mask = torch.zeros(len(joints_name), 1).bool()
    for sh in cfg['shoulders']:
        sh_mask[joints_index[sh]] = True

    # elbow mask
    el_mask = torch.zeros(len(joints_name), 1).bool()
    for el in cfg['elbows']:
        el_mask[joints_index[el]] = True

    # node parent
    parent = -torch.ones(len(joints_name)).long()
    for edge in edge_index.permute(1, 0):
        parent[edge[1]] = edge[0]

    # node offset
    offset = torch.stack([torch.Tensor(joints[joint]['origin']) for joint in joints_name], dim=0)
    # change root offset to store init pose
    init_pose = {}
    fk = robot.link_fk()
    for link, matrix in fk.items():
        init_pose[link.name] = matrix_to_xyz_rpy(matrix)
    origin = torch.zeros(6)
    for root in cfg['root_name']:
        offset[joints_index[root]] = torch.Tensor(init_pose[joints[root]['child']])
        origin[:3] += offset[joints_index[root]][:3]
    origin /= 2
    # move relative to origin
    for root in cfg['root_name']:
        offset[joints_index[root]] -= origin
    # print(offset, offset.shape)

    # dist to root
    root_dist = torch.zeros(len(joints_name), 1)
    for node_idx in range(len(joints_name)):
        dist = 0
        current_idx = node_idx
        while current_idx != -1:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        root_dist[node_idx] = dist
    print(root_dist, root_dist.shape)
    # 在 yumi2graph 函数中添加
    # shoulder_map = {0:1,1: 1, 2: 1, 3: 1, 4: 1,5:6, 6: 6, 7: 6, 8: 6, 9: 6}
    #     # 替换原来的 shoulder_dist 循环部分
    # shoulder_dist = torch.zeros(len(joints_name), 1)
    # for node_idx in range(len(joints_name)):
    #     if node_idx in shoulder_map:
    #         shoulder_idx = shoulder_map[node_idx]
    #         shoulder_dist[node_idx] = abs(root_dist[node_idx] - root_dist[shoulder_idx])
    #     else:
    #         # 可选：设置 NaN 或者 0 表示不属于任何肩部链路
    #         shoulder_dist[node_idx] = float('nan')

    

    # elbow_map = {0:3,1:3,2:3,3: 3, 4: 3,5: 8,6: 8,7: 8,8: 8, 9: 8}
    # # 替换原来的 elbow_dist 循环部分
    # # 计算每个节点到肘部的距离
    # elbow_dist = torch.zeros(len(joints_name), 1)
    # for node_idx in range(len(joints_name)):
    #     if node_idx in elbow_map:
    #         elbow_idx = elbow_map[node_idx]
    #         elbow_dist[node_idx] =abs( root_dist[node_idx] - root_dist[elbow_idx])
    #     else:
    #         # 可选：设置 NaN 或者 0 表示不属于任何肘部链路
    #         elbow_dist[node_idx] = float('nan')

    # dist to shoulder
    shoulder_dist = torch.zeros(len(joints_name), 1)
    for node_idx in range(len(joints_name)):
        dist = 0
        current_idx = node_idx
        while current_idx != -1 and joints_name[current_idx] not in cfg['shoulders']:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        shoulder_dist[node_idx] = dist
    print(shoulder_dist, shoulder_dist.shape)
    # 旧的计算肘部距离的方式
    # dist to elbow
    elbow_dist = torch.zeros(len(joints_name), 1)
    for node_idx in range(len(joints_name)):
        dist = 0
        current_idx = node_idx
        while current_idx != -1 and joints_name[current_idx] not in cfg['elbows']:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        elbow_dist[node_idx] = dist
    print(elbow_dist, elbow_dist.shape)

    # rotation axis
    axis = [torch.Tensor(joints[joint]['axis']) for joint in joints_name]
    axis = torch.stack(axis, dim=0)

    # joint limit
    lower = [torch.Tensor([joints[joint]['lower']]) for joint in joints_name]
    lower = torch.stack(lower, dim=0)
    upper = [torch.Tensor([joints[joint]['upper']]) for joint in joints_name]
    upper = torch.stack(upper, dim=0)
    # print(lower.shape, upper.shape)

    # skeleton
    data = Data(x=torch.zeros(num_nodes, 1),
                edge_index=edge_index,
                edge_attr=edge_attr,
                skeleton_type=skeleton_type,
                topology_type=topology_type,
                ee_mask=ee_mask,
                sh_mask=sh_mask,
                el_mask=el_mask,
                root_dist=root_dist,
                shoulder_dist=shoulder_dist,
                elbow_dist=elbow_dist,
                num_nodes=num_nodes,
                parent=parent,
                offset=offset,
                axis=axis,
                lower=lower,
                upper=upper
                )

    # test forward kinematics
    # print(joints_name)
    # result = robot.link_fk(cfg={joint:0.0 for joint in joints_name})
    # for link, matrix in result.items():
        # print(link.name, matrix)
    # import os, sys, inspect
    # currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # parentdir = os.path.dirname(currentdir)
    # sys.path.insert(0,parentdir)

    from models.kinematics import ForwardKinematicsURDF
    import matplotlib.pyplot as plt
    show_rotation_axes = True
    fk = ForwardKinematicsAxis()
    #[1]要反转，[2]要反转 [4]要反转 [5]要反转!!! 90 90 -90 90 90 90
    # angle_x = torch.tensor([[90 ,-90 ,90 ,90,-90 ,-90 ,-49.0 ,-51.0 ,-32.0 ,84.0,-53.0,104.0]])
    # angle_x = angle_x/180.0*3.14
    pos, rot, pos1= fk(angle_x, data.parent, data.offset, 1, data.axis)
    
    if figure_window is None:
       import matplotlib.pyplot as plt
       fig = plt.figure()
    else :
       fig = figure_window
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-0.3,0.5)
    ax.set_ylim3d(-0.5,0.5)
    ax.set_zlim3d(-0.3,0.3)
    
    # plot 3D lines
    for edge in edge_index.permute(1, 0):
        line_x = [pos[edge[0]][0], pos[edge[1]][0]]
        line_y = [pos[edge[0]][1], pos[edge[1]][1]]
        line_z = [pos[edge[0]][2], pos[edge[1]][2]]
        plt.plot(line_x, line_y, line_z, 'royalblue', marker='o')
        
    # 在每个关节上标注 index
    for i, joint in enumerate(pos):
        x, y, z = joint.tolist()
        ax.text(x, y, z, f'{i}', fontsize=10, color='black')
        
    # 绘制旋转坐标轴
    if show_rotation_axes:
        for i in range(len(pos)):
            draw_coordinate_axes(ax, pos[i].cpu().numpy(), rot[i], scale=0.05)
    
    plt.show()
    return data


"""
convert Inspire Hand URDF graph
"""
def hand2graph(urdf_file, cfg):
    # load URDF
    robot = URDF.load(urdf_file)

    # parse joint params
    joints = {}
    for joint in robot.joints:
        # joint atributes
        joints[joint.name] = {'type': joint.joint_type, 'axis': joint.axis,
                              'parent': joint.parent, 'child': joint.child,
                              'origin': matrix_to_xyz_rpy(joint.origin),
                              'lower': joint.limit.lower if joint.limit else 0,
                              'upper': joint.limit.upper if joint.limit else 0}


    # debug msg
    # for name, attr in joints.items():
    #     print(name, attr)

    # skeleton type & topology type
    skeleton_type = 0
    topology_type = 0

    # collect edge index & edge feature
    joints_name = cfg['joints_name']
    joints_index = {name: i for i, name in enumerate(joints_name)}
    edge_index = []
    edge_attr = []
    for edge in cfg['edges']:
        parent, child = edge
        # print(edge)
        # print(joints)
        # print(22222)
        # add edge index
        edge_index.append(torch.LongTensor([joints_index[parent], joints_index[child]]))
        # add edge attr
        edge_attr.append(torch.Tensor(joints[child]['origin']))
    edge_index = torch.stack(edge_index, dim=0)
    edge_index = edge_index.permute(1, 0)
    edge_attr = torch.stack(edge_attr, dim=0)
    # print(edge_index, edge_attr, edge_index.shape, edge_attr.shape)

    # number of nodes
    num_nodes = len(joints_name)
    # print(num_nodes)

    # end effector mask
    ee_mask = torch.zeros(len(joints_name), 1).bool()
    for ee in cfg['end_effectors']:
        ee_mask[joints_index[ee]] = True
    # print(ee_mask)

    # elbow mask
    el_mask = torch.zeros(len(joints_name), 1).bool()
    for el in cfg['elbows']:
        el_mask[joints_index[el]] = True
    # print(el_mask)

    # nv mask
    nv_mask = torch.zeros(len(joints_name), 1).bool()
    for nv in cfg['norm_vector']:
        nv_mask[joints_index[nv]] = True
    # print(el_mask)

    # node parent
    parent = -torch.ones(len(joints_name)).long()
    for edge in edge_index.permute(1, 0):
        parent[edge[1]] = edge[0]
    # print(parent)

    # node offset
    offset = []
    for joint in joints_name:
        offset.append(torch.Tensor(joints[joint]['origin']))
    offset = torch.stack(offset, dim=0)
    # print(offset, offset.shape)

    # dist to root
    root_dist = torch.zeros(len(joints_name), 1)
    for node_idx in range(len(joints_name)):
        dist = 0
        current_idx = node_idx
        while joints_name[current_idx] != cfg['root_name']:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        root_dist[node_idx] = dist
    # print(root_dist, root_dist.shape)

    # dist to elbow
    elbow_dist = torch.zeros(len(joints_name), 1)
    for node_idx in range(len(joints_name)):
        dist = 0
        current_idx = node_idx
        while joints_name[current_idx] != cfg['root_name'] and joints_name[current_idx] not in cfg['elbows']:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        elbow_dist[node_idx] = dist
    # print(elbow_dist, elbow_dist.shape)

    # rotation axis
    axis = [torch.Tensor(joints[joint]['axis']) if joint != cfg['root_name'] else torch.zeros(3) for joint in joints_name]
    axis = torch.stack(axis, dim=0)
    # print(axis, axis.shape)

    # joint limit
    lower = [torch.Tensor([joints[joint]['lower']]) if joint != cfg['root_name'] else torch.zeros(1) for joint in joints_name]
    lower = torch.stack(lower, dim=0)
    upper = [torch.Tensor([joints[joint]['upper']]) if joint != cfg['root_name'] else torch.zeros(1) for joint in joints_name]
    upper = torch.stack(upper, dim=0)
    #print(lower, upper, lower.shape, upper.shape)

    # skeleton
    data = Data(x=torch.zeros(num_nodes, 1),
                edge_index=edge_index,
                edge_attr=edge_attr,
                skeleton_type=skeleton_type,
                topology_type=topology_type,
                ee_mask=ee_mask,
                el_mask=el_mask,
                nv_mask=nv_mask,
                root_dist=root_dist,
                elbow_dist=elbow_dist,
                num_nodes=num_nodes,
                parent=parent,
                offset=offset,
                axis=axis,
                lower=lower,
                upper=upper)
    # data for arm with hand
    data.hand_x = data.x
    data.hand_edge_index = data.edge_index
    data.hand_edge_attr = data.edge_attr
    data.hand_ee_mask = data.ee_mask
    data.hand_el_mask = data.el_mask
    data.hand_ee_mask = data.ee_mask
    data.hand_nv_mask = data.nv_mask
    data.hand_root_dist = data.root_dist
    data.hand_elbow_dist = data.elbow_dist
    data.hand_num_nodes = data.num_nodes
    data.hand_parent = data.parent
    data.hand_offset = data.offset
    data.hand_axis = data.axis
    data.hand_lower = data.lower
    data.hand_upper = data.upper
    # print(data)

    # # test forward kinematics
    result = robot.link_fk(cfg={joint: 0.0 for joint in cfg['joints_name'] if joint != cfg['root_name']})
    # for link, matrix in result.items():
    #     print(link.name, matrix)
    fk = ForwardKinematicsAxis()
    pos, rot, _ = fk(data.x, data.parent, data.offset, 1, data.axis)
    # print(rot)
    # print(joints_index, pos)
    # print(pos)

    # # visualize
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_axis_off()
    # ax.view_init(elev=0, azim=90)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-0.2,0.2)
    ax.set_ylim3d(-0.1,0.1)
    ax.set_zlim3d(-0.1,0.1)
    
    # plot 3D lines
    for edge in edge_index.permute(1, 0):
        line_x = [pos[edge[0]][0], pos[edge[1]][0]]
        line_y = [pos[edge[0]][1], pos[edge[1]][1]]
        line_z = [pos[edge[0]][2], pos[edge[1]][2]]
        # line_x = [pos[edge[0]][2], pos[edge[1]][2]]
        # line_y = [pos[edge[0]][0], pos[edge[1]][0]]
        # line_z = [pos[edge[0]][1], pos[edge[1]][1]]
        plt.plot(line_x, line_y, line_z, 'royalblue', marker='o')
    plt.show()
    # plt.savefig('hand.png')

    return data

if __name__ == '__main__':

    yumi_cfg = {
    'joints_name': [
        'J11',  # L
        'J12',
        'J13',
        'J14',
        'J15',
        'J16',

        'J5',  # R
        'J6',
        'J7',
        'J8',
        'J9',
        'J10',
    ],
    'edges': [
        ['J11', 'J12'],
        ['J12', 'J13'],
        ['J13', 'J14'],
        ['J14', 'J15'],
        ['J15', 'J16'],

        ['J5', 'J6'],
        ['J6', 'J7'],
        ['J7', 'J8'],
        ['J8', 'J9'],
        ['J9', 'J10'],
    ],
    'root_name': [
        'J11',
        'J5',
    ],
    'end_effectors': [
        'J16',
        'J10',
    ],
    'shoulders': [
        'J12',
        'J6',
    ],
    'elbows': [
        'J13',
        'J7',
    ],
    }
    graph = yumi2graph(urdf_file="D:\\2025\\crp\\ur3_robot_train_pico\\data\\target\\ur3\\robot(ur3).urdf", cfg=yumi_cfg)
    # print(graph.lower.size())
    print('yumi', graph)
    #
    # hand_cfg = {
    #     'joints_name': [
    #         'yumi_link_7_r_joint',
    #         'Link1',
    #         'Link11',
    #         'R_ring_tip_joint',

    #         'Link2',
    #         'Link22',
    #         'R_middle_tip_joint',

    #         'Link3',
    #         'Link33',
    #         'R_index_tip_joint',

    #         'Link4',
    #         'Link44',
    #         'R_pinky_tip_joint',

    #         'Link5',
    #         'Link51',
    #         'Link52',
    #         'Link53',
    #         'R_thumb_tip_joint',
    #     ],
    #     'edges': [
    #         ['yumi_link_7_r_joint', 'Link1'],
    #         ['Link1', 'Link11'],
    #         ['Link11', 'R_ring_tip_joint'],
    #         ['yumi_link_7_r_joint', 'Link2'],
    #         ['Link2', 'Link22'],
    #         ['Link22', 'R_middle_tip_joint'],
    #         ['yumi_link_7_r_joint', 'Link3'],
    #         ['Link3', 'Link33'],
    #         ['Link33', 'R_index_tip_joint'],
    #         ['yumi_link_7_r_joint', 'Link4'],
    #         ['Link4', 'Link44'],
    #         ['Link44', 'R_pinky_tip_joint'],
    #         ['yumi_link_7_r_joint', 'Link5'],
    #         ['Link5', 'Link51'],
    #         ['Link51', 'Link52'],
    #         ['Link52', 'Link53'],
    #         ['Link53', 'R_thumb_tip_joint'],
    #     ],
    #     'root_name': 'yumi_link_7_r_joint',
    #     'end_effectors': [
    #         'R_index_tip_joint',
    #         'R_middle_tip_joint',
    #         'R_ring_tip_joint',
    #         'R_pinky_tip_joint',
    #         'R_thumb_tip_joint',
    #     ],
    #     # 'end_effectors': [
    #     #     'Link11',
    #     #     'Link22',
    #     #     'Link33',
    #     #     'Link44',
    #     #     'Link53',
    #     # ],
    #     'elbows': [
    #         'Link1',
    #         'Link2',
    #         'Link3',
    #         'Link4',
    #         'Link5',
    #     ],
    #     'norm_vector': [
    #         'yumi_link_7_r_joint',
    #         'Link1',
    #         'Link3',
    #     ],
    # }
    # graph = hand2graph(urdf_file="D:\\2025\\crp\\ur3_robot_train_pico\\data\\target\\ur3\\robot(ur3).urdf", cfg=hand_cfg)
    # print('hand', graph)
