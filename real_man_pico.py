import sys
import time

import xrobotoolkit_sdk as xrt
from multiprocessing import Manager, Process

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#重定义引用库
import os
import copy
import logging
import argparse
from tensorboardX import SummaryWriter
#from Robotic_Arm import rm_robot_interface as rm


from models import model
from models.loss import CollisionLoss, JointLimitLoss, RegLoss, calculate_all_loss
from models import model_ur3
from mylib.pico2real import *
from train import train_epoch

from utils.config import cfg
from utils.util import create_folder

import gym, yumi_gym
import pybullet as p

import torch
import torch.nn as nn
import torch_geometric.transforms as transforms
import torch.optim as optim
from torch_geometric.data import Batch, DataListLoader


from datetime import datetime
import dataset
from dataset import Normalize,parse_arm_realtime
#关节名称对应字典
'''
名称对应0: Pelvis, 1: Left Hip, 2: Right Hip, 3: Spine1, 4: Left Knee, 5: Right Knee
    6: Spine2, 7: Left Ankle, 8: Right Ankle, 9: Spine3, 10: Left Foot, 11: Right Foot
    12: Neck, 13: Left Collar, 14: Right Collar, 15: Head, 16: Left Shoulder, 17: Right Shoulder
    18: Left Elbow, 19: Right Elbow, 20: Left Wrist, 21: Right Wrist, 22: Left Hand, 23: Right Hand,24:timestamp
    '''
joint_names = [
    "Pelvis", "Left Hip", "Right Hip", "Spine1", "Left Knee", "Right Knee",
    "Spine2", "Left Ankle", "Right Ankle", "Spine3", "Left Foot", "Right Foot",
    "Neck", "Left Collar", "Right Collar", "Head", "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hand", "Right Hand"
]

# 需要保存的12个关节库对应关系
selected_joints = {
    "l_up": 16,  # Left Shoulder
    "r_up": 17,  # Right Shoulder
    "l_fr": 18,  # Left Elbow
    "r_fr": 19,  # Right Elbow
    "l_hd": 20,  # Left Wrist
    "r_hd": 21,  # Right Wrist
    "l_hand": 22,  # Left Hand
    "r_hand": 23,  # Right Hand
}

def run_data_tests(robotic_data):
    '''
    数据获取进程
    save为True时保存数据，为False时不保存数据
    保存数据中，body为24*7的数据，故建24个库，对应24个关节点，加入一个库为时间戳，然后将数据逐行储存，保存为h5文件
    名称对应0: Pelvis, 1: Left Hip, 2: Right Hip, 3: Spine1, 4: Left Knee, 5: Right Knee
    6: Spine2, 7: Left Ankle, 8: Right Ankle, 9: Spine3, 10: Left Foot, 11: Right Foot
    12: Neck, 13: Left Collar, 14: Right Collar, 15: Head, 16: Left Shoulder, 17: Right Shoulder
    18: Left Elbow, 19: Right Elbow, 20: Left Wrist, 21: Right Wrist, 22: Left Hand, 23: Right Hand,24:timestamp
    '''
    save=True
    filename='operating.h5'
    frame_num = 0
    print("Starting Python binding test...")
    if save:
        #使用h5py保存数据，先新建一个文件
        import h5py
        h5f = h5py.File(filename,'w')
        # 新建24个库
        # 新建24个库，指定数据类型和可变长度
        for name in joint_names:
            h5f.create_dataset(name, (0, 7), maxshape=(None, 7), dtype='f')
        h5f.create_dataset('timestamp', (0,), maxshape=(None,), dtype='f')
    print("Initializing SDK...")
    xrt.init()
    print("SDK Initialized successfully.")
    blankframe = 0
    try:
        while not robotic_data['EXIT']:
            print(f"\n--- Iteration {frame_num} ---")
            frame_num = frame_num + 1
            #检测停止按钮
            if xrt.get_A_button():
                break
            if xrt.get_left_trigger():
                #反转抓取符号
                robotic_data['left_grasp'] = not robotic_data['left_grasp']
            if xrt.get_right_trigger():
                #反转抓取符号
                robotic_data['right_grasp'] = not robotic_data['right_grasp']
            # 处理选定关节数据，转换为 dataset.py 可解析的格式
            processed_data = {}
            # Check if body tracking data is available
            if xrt.is_body_data_available():
                blankframe = 0
                # Get body joint poses (24 joints, 7 values each: x,y,z,qx,qy,qz,qw)
                body_poses = xrt.get_body_joints_pose()
                robotic_data['body'] = body_poses
                # print(f"Body joints pose data: {body_poses}")
                # print('********上面为机器人')
                if save:
                    # 保存数据到h5文件
                    for name in joint_names:
                        #获取名字对应索引
                        index = joint_names.index(name)
                        #调整数据集大小并添加新数据
                        dataset = h5f[name]
                        dataset.resize((dataset.shape[0] + 1, dataset.shape[1]))
                        #将数据写入对应的库的新一行
                        dataset[-1, :] = body_poses[index]
                # 处理每个选定关节的位置和四元数
                for joint_name, joint_index in selected_joints.items():
                    # 获取关节数据 (x, y, z, qx, qy, qz, qw)
                    joint_data = body_poses[joint_index]
                    
                    # 分离位置和四元数
                    position = joint_data[:3]
                    quaternion = joint_data[3:]
                    
                    # 对位置进行坐标系转换
                    transformed_position = trans_xyz(np.array([position]))[0]
                    
                    # 对四元数进行坐标系转换
                    transformed_quaternion = trans_quat(quaternion)
                    
                    # 保存转换后的数据
                    processed_data[f'{joint_name}_pos'] = transformed_position
                    processed_data[f'{joint_name}_quat'] = transformed_quaternion
                
                # 获取左手和右手的四元数数据
                l_hand_data = body_poses[selected_joints["l_hand"]]
                r_hand_data = body_poses[selected_joints["r_hand"]]
                
                l_hand_quat = l_hand_data[3:]
                r_hand_quat = r_hand_data[3:]
                
                # 对四元数进行坐标系转换
                transformed_l_hand_quat = trans_quat(l_hand_quat)
                transformed_r_hand_quat = trans_quat(r_hand_quat)
                
                # 计算掌心向量
                l_palm_vector = get_palm_vector(transformed_l_hand_quat)
                r_palm_vector = get_palm_vector(transformed_r_hand_quat)
                
                # 保存掌心向量
                processed_data['l_hd_vec'] = l_palm_vector
                processed_data['r_hd_vec'] = r_palm_vector
                robotic_data['processed_data'] = processed_data
            else:
                blankframe = blankframe +1
                print("Body tracking data not available.")
            if blankframe > 1000:
                    break
            time.sleep(0.02)  # Wait for 0.01 seconds before the next iteration
        
        robotic_data['EXIT'] = True
        print("\nAll iterations complete.")
        if save:
            h5f.close()
    except Exception as e:
        print("数据获取单元出错")
    finally:
        print("\nClosing SDK...")
    xrt.close()
    print("SDK closed.")
    print("Test finished.")
    robotic_data['EXIT'] = True

def show_data(robotic_data):
    # '''
    # 显示数据，定时刷新
    # '''
    # # 创建3D图形窗口
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # # 设置坐标轴范围和标签
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    # ax.set_zlim(0, 3)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Real-time Body Tracking Data')
    
    # while not robotic_data['EXIT']:
    #     # 清除之前的点
    #     ax.cla()
    #     # 重新设置坐标轴范围和标签
    #     ax.set_xlim(-2, 2)
    #     ax.set_ylim(-2, 2)
    #     ax.set_zlim(0, 3)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_title('Real-time Body Tracking Data')
        
    #     # 获取身体位置数据
    #     body_data = robotic_data.get('body')
        
    #     if body_data is not None:
    #         # 提取所有关节的位置坐标(x, y, z)
    #         positions = np.array([[joint[0], joint[1], joint[2]] for joint in body_data])
            
    #         # 绘制所有关节
    #         xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
    #         ax.scatter(xs, ys, zs, c='r', marker='o')
            
    #         # 为每个关节点添加标签
    #         for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
    #             ax.text(x, y, z, f'{joint_names[i]}', fontsize=8)
                
    #     # 刷新图形
    #     plt.pause(0.1)
        
    # plt.close(fig)

    from urdf2graph import yumi2graph
    import matplotlib.pyplot as plt
    fig = plt.figure()
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
    while not robotic_data['EXIT']:
       if 'retargeting_output' in robotic_data and robotic_data['retargeting_output'] is not None:
           angles = robotic_data['retargeting_output']
           #将angles转为tensor
           angles = torch.tensor(angles)
def retargeting_process(robotic_data):
    '''
    调用模型进行实时重定向
    '''
    #初始化
    # Argument parse
    while not robotic_data['simulation_started']:
        time.sleep(0.1)
    parser = argparse.ArgumentParser(description='Inference with trained model')
    parser.add_argument('--cfg', default='configs/inference/arm.yaml', type=str, help='Path to configuration file')
    args = parser.parse_args()

    # Configurations parse
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    # print(cfg)

    # Create folder
    create_folder(cfg.OTHERS.LOG)
    create_folder(cfg.OTHERS.SUMMARY)
    # Create logger & tensorboard writer
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.FileHandler(os.path.join(cfg.OTHERS.LOG, "{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now()))), logging.StreamHandler()])
    logger = logging.getLogger()
    writer = SummaryWriter(os.path.join(cfg.OTHERS.SUMMARY, "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())))
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_transform = transforms.Compose([Normalize()])
    
    # 初始化损失函数和优化器
    # end effector loss
    ee_criterion = nn.MSELoss() if cfg.LOSS.EE else None
    # vector similarity loss
    vec_criterion = nn.MSELoss() if cfg.LOSS.VEC else None
    # collision loss
    col_criterion = CollisionLoss(cfg.LOSS.COL_THRESHOLD) if cfg.LOSS.COL else None
    # joint limit loss
    lim_criterion = JointLimitLoss() if cfg.LOSS.LIM else None
    # end effector orientation loss
    ori_criterion = nn.MSELoss() if cfg.LOSS.ORI else None
    # finger similarity loss
    fin_criterion = nn.MSELoss() if cfg.LOSS.FIN else None
    # regularization loss
    reg_criterion = RegLoss() if cfg.LOSS.REG else None
    # 加载机器人
    test_target = sorted([target for target in getattr(dataset, cfg.DATASET.TEST.TARGET_NAME)(
                root=cfg.DATASET.TEST.TARGET_PATH)], key=lambda target: target.skeleton_type)
    # 加载模型
    model = getattr(model_ur3, cfg.MODEL.NAME)().to(device)
    # Load checkpoint
    if cfg.MODEL.CHECKPOINT is not None:
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))
    else:
        #报错 没有模型文件
        assert('没有模型文件')
    # 准备位置和四元数数据字典
    pos_body = {}
    q_body = {}
    robot_angle = []
    print("模型初始化完毕")
    while not robotic_data['EXIT']:
        t0=time.time()
        # 检查是否有新的处理数据
        if 'processed_data' in robotic_data and robotic_data['processed_data'] is not None:
            processed_data = robotic_data['processed_data']
            # print(processed_data)
            # 提取位置数据
            pos_body['L_Shoulder'] = processed_data['l_up_pos']
            pos_body['R_Shoulder'] = processed_data['r_up_pos']
            pos_body['L_Elbow'] = processed_data['l_fr_pos']
            pos_body['R_Elbow'] = processed_data['r_fr_pos']
            pos_body['L_Wrist'] = processed_data['l_hd_pos']
            pos_body['R_Wrist'] = processed_data['r_hd_pos']
            # 提取四元数数据
            q_body['L_Shoulder'] = processed_data['l_up_quat']
            q_body['R_Shoulder'] = processed_data['r_up_quat']
            q_body['L_Elbow'] = processed_data['l_fr_quat']
            q_body['R_Elbow'] = processed_data['r_fr_quat']
            q_body['L_Wrist'] = processed_data['l_hd_quat']
            q_body['R_Wrist'] = processed_data['r_hd_quat']
            
            # 获取手掌向量
            l_hand_vec = processed_data['l_hd_vec']
            r_hand_vec = processed_data['r_hd_vec']
            
            # 使用 parse_arm_realtime 函数处理数据
            try:
                test_data = parse_arm_realtime(None, None, pos_body, q_body, l_hand_vec, r_hand_vec)
                # 处理好的数据放入 robotic_data 供其他进程使用
                # robotic_data['retargeting_input'] = data_list
                # TODO: 在这里调用模型进行推理
                test_data = [pre_transform(data) for data in test_data]
                test_loader = [test_data]
                model.eval()
                z_all = []
                # 这里应该将 data_list 传递给模型进行重定向计算
                print("数据已处理并准备好用于重定向")
                for batch_idx, data_list in enumerate(test_loader):
                        for target_idx, target in enumerate(test_target):
                            z = model.encode(Batch.from_data_list(data_list).to(device)).detach()
                            z.requires_grad = True
                            z_all.append(z)
                # Create optimizer
                optimizer = optim.Adam(z_all, lr=cfg.HYPER.LEARNING_RATE)
                best_loss = 10000
                best_z_all = copy.deepcopy(z_all)
                best_cnt = 0
                for epoch in range(cfg.HYPER.EPOCHS):
                        train_loss = train_epoch(model, ee_criterion, vec_criterion, col_criterion, lim_criterion,
                                                ori_criterion, fin_criterion, reg_criterion,0, optimizer, test_loader,
                                                test_target, epoch, logger, cfg.OTHERS.LOG_INTERVAL, writer, device,
                                                z_all)
                        # if cfg.INFERENCE.MOTION.KEY:
                        # Save model
                        if train_loss > best_loss:
                            best_cnt += 1
                        else:
                            best_cnt = 0
                            best_loss = train_loss
                            best_z_all = copy.deepcopy(z_all)
                        if best_cnt == 5:
                            logger.info("Interation Finished")
                            break
                print('重定向损失',best_loss)
                # store final results
                model.eval()
                pos_all = []
                ang_all = []
            
                for batch_idx, data_list in enumerate(test_loader):
                    for target_idx, target in enumerate(test_target):
                        # fetch target
                        target_list = [target for data in data_list]
                        # fetch z
                        z = best_z_all[batch_idx]
                        # forward
                        z, ang, target_pos, target_rot, global_pos, l_hand_ang, l_hand_pos, r_hand_ang,r_hand_pos = model.decode(z, Batch.from_data_list(target_list).to(z.device))
                        loss = calculate_all_loss(data_list, target_list, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, fin_criterion, reg_criterion,0,
                                            z, ang, target_pos, target_rot, global_pos, l_hand_pos, r_hand_pos)
                        if ang is not None:
                            robot_angle = ang
                            # print('重定向输出',ang)
                            robotic_data['retargeting_output'] = ang.detach().cpu().numpy()
                            # ang_all.append(ang)
                        # if global_pos is not None:
                            # pos_all.append(global_pos)
            except Exception as e:
                print(f"处理数据时出错: {e}")
        t1 = time.time() - t0
        print(f"处理数据用时 {t1:.2f} 秒")
        time.sleep(0.001)

def control_robot(robotic_data):
    '''
    机器人控制进程
    '''
    '''
    机器人控制进程
    '''
    try:
        # 初始化机械臂（这里使用与 realtime_robot.py 类似的接口
        # 创建机械臂实例
        l_arm = rm.RoboticArm(rm.rm_thread_mode_e.RM_TRIPLE_MODE_E)
        l_handle = l_arm.rm_create_robot_arm("192.168.10.19", 8080)
        r_arm = rm.RoboticArm(rm.rm_thread_mode_e.RM_TRIPLE_MODE_E)
        r_handle = r_arm.rm_create_robot_arm("192.168.10.18", 8080)
        
        print("机械臂连接成功, 左臂ID:", l_handle.id, "右臂ID:", r_handle.id)
        
        # 定义关节限制
        min_angles = [-178.0, -130.0, -135.0, -180.0, -128.0, -360.0]
        max_angles = [178.0, 130.0, 135.0, 180.0, 128.0, 360.0]
        left_hand_grasp_temp = False
        right_hand_grasp_temp = False
        # 初始化位置
        init_angles = [0.0, -90, 0, 0.0, 0.0, 0.0]
        l_arm.rm_movej(init_angles, 50, 0, 0, 1)
        r_arm.rm_movej(init_angles, 50, 0, 0, 1)
        
        print("机械臂已初始化到初始位置")
        robotic_data['simulation_started'] = True
        while not robotic_data['EXIT']:
            # 检查是否有重定向输出数据
            if 'retargeting_output' in robotic_data and robotic_data['retargeting_output'] is not None:
                try:
                    # 获取重定向输出的角度数据
                    angles = robotic_data['retargeting_output']
                    left_hand_grasp = robotic_data['left_grasp']
                    right_hand_grasp = robotic_data['right_grasp']

                    # 如果是tensor类型，转换为numpy数组
                    if hasattr(angles, 'detach'):
                        angles = angles.detach().cpu().numpy()
                    
                    # 展平数组
                    angles = angles.flatten()
                    # print(f"接收到重定向角度数据: {angles}")
                    # 分离左右臂角度（前6个是左臂，后6个是右臂）
                    if len(angles) >= 12:
                        l_joint_angles = angles[0:6]
                        r_joint_angles = angles[6:12]
                        if left_hand_grasp != left_hand_grasp_temp:
                            if left_hand_grasp:
                                print(l_arm.rm_set_hand_angle([100,100,100,100,100,100],True,1))
                            elif ~left_hand_grasp:
                                print(l_arm.rm_set_hand_angle([800,800,800,800,800,800],True,1))
                            time.sleep(0.2)
                        if right_hand_grasp != right_hand_grasp_temp:
                            if right_hand_grasp:
                                print(r_arm.rm_set_hand_angle([100,100,100,100,100,100],True,1))
                            elif ~right_hand_grasp:
                                print(r_arm.rm_set_hand_angle([800,800,800,800,800,800],True,1))
                            time.sleep(0.2)
                        left_hand_grasp_temp = left_hand_grasp
                        right_hand_grasp_temp = right_hand_grasp
                        # 转换为实际机器人坐标系
                        l_robot_angles = trans2realworld1(l_joint_angles, min_angles, max_angles)
                        r_robot_angles = trans2realworld(r_joint_angles, min_angles, max_angles)
                        
                        # print(f"左臂角度: {l_robot_angles}")
                        # print(f"右臂角度: {r_robot_angles}")
                        
                        # 发送命令到机械臂
                        # 左臂控制
                        l_result = l_arm.rm_movej_canfd(l_robot_angles, False, 0, 1, 50)
                        
                        
                        # 右臂控制
                        r_result = r_arm.rm_movej_canfd(r_robot_angles, False, 0, 1, 50)
                        print(f"左臂控制结果: {l_result}","右臂控制结果: {r_result}")
                    
                    # 短暂延时
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"控制机械臂时出错: {e}")
                    time.sleep(0.1)
            else:
                # 没有数据时短暂等待
                time.sleep(0.01)
                
    except Exception as e:
        print(f"机器人控制进程出错: {e}")
        robotic_data['EXIT'] = True
    finally:
        # 程序结束时回到初始位置
        robotic_data['EXIT'] = True
        try:
            print("程序结束，机械臂回到初始位置...")
            init_angles = [0.0, -90, 0, 0.0, 0.0, 0.0]
            l_arm.rm_movej(init_angles, 50, 0, 0, 1)
            r_arm.rm_movej(init_angles, 50, 0, 0, 1)
            # 断开连接
            rm.RoboticArm.rm_destroy()
            print("机械臂连接已断开")
        except:
            pass

def simulation_process(robotic_data):
    '''
    虚拟仿真进程
    '''
    try:
        # 初始化仿真环境
        body_ang = np.zeros(12)
        env = gym.make('yumi-v0')
        observation = env.reset()

        # 设置相机参数
        camera_distance = 2
        camera_yaw = 140
        camera_pitch = -15
        camera_target_position = [0, 0.2, 0.3]

        print("虚拟仿真平台已初始化")
        robotic_data['simulation_started'] = True
        while not robotic_data['EXIT']:
            # 检查是否有重定向输出数据
            if 'retargeting_output' in robotic_data and robotic_data['retargeting_output'] is not None:
                try:
                    # 获取重定向输出的角度数据
                    angles = robotic_data['retargeting_output']
                    
                    # 如果是tensor类型，转换为numpy数组
                    if hasattr(angles, 'detach'):
                        angles = angles.detach().cpu().numpy()
                    
                    # 展平数组
                    angles = angles.flatten()
                    
                    # 分离左右臂角度（前6个是左臂，后6个是右臂）
                    if len(angles) != 12:
                        body_ang = angles[0:12]
                        print('数据异常,应为12')
                    else :
                        body_ang = angles
                    # 渲染环境
                    env.render()
                    p.resetDebugVisualizerCamera(
                        cameraDistance=camera_distance,
                        cameraYaw=camera_yaw,
                        cameraPitch=camera_pitch,
                        cameraTargetPosition=camera_target_position
                    )
                    
                    # 构造动作向量（只控制手臂，不控制手部）
                    action = body_ang
                    
                    # 执行动作
                    observation, reward, done, info = env.step(action)
                    
                    # # 处理键盘输入来控制相机视角
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
                    #         elif k == ord('j'):
                    #             camera_target_position[0] -= 0.3
                    #         elif k == ord('l'):
                    #             camera_target_position[0] += 0.3
                    #         elif k == ord('i'):
                    #             camera_target_position[1] += 0.3
                    #         elif k == ord('k'):
                    #             camera_target_position[1] -= 0.3
                    #         elif k == ord('u'):
                                # camera_target_position[2] += 0.3
                            # elif k == ord('o'):
                                # camera_target_position[2] -= 0.3
                    
                    time.sleep(0.001)  # 控制仿真速度
                    
                except Exception as e:
                    print(f"虚拟仿真过程中出错: {e}")
                    time.sleep(0.05)
            else:
                # 没有数据时短暂等待
                time.sleep(0.01)
                
    except Exception as e:
        print(f"虚拟仿真进程出错: {e}")
        robotic_data['EXIT'] = True
    finally:
        robotic_data['EXIT'] = True
        try:
            # 关闭环境
            env.close()
        except:
            pass
        print("虚拟仿真平台已关闭")

if __name__ == "__main__":
        # 创建进程间通信管理器
    robot_manager = Manager()
    # 共享字典用于进程间数据传递
    robotic_data = robot_manager.dict()
    robotic_data['body'] = None
    robotic_data['head'] = None
    robotic_data['pose_l'] = None
    robotic_data['pose_r'] = None
    robotic_data['EXIT'] = False
    robotic_data['simulation_started'] = False
    #右手抓取信号
    robotic_data['right_grasp'] = False
    #左手抓取信号
    robotic_data['left_grasp'] = False
    get_data_process = Process(target=run_data_tests, args=(robotic_data,))
    get_data_process.daemon = True
    get_data_process.start()

    show_data_process = Process(target=show_data, args=(robotic_data,))
    show_data_process.daemon = True
    show_data_process.start()

    retargeting_data_process = Process(target=retargeting_process, args=(robotic_data,))
    retargeting_data_process.daemon = True
    retargeting_data_process.start()
    
    simulation_data_process = Process(target=simulation_process, args=(robotic_data,))
    # simulation_data_process = Process(target=control_robot, args=(robotic_data,))
    simulation_data_process.daemon = True
    simulation_data_process.start()

    while True:
        time.sleep(0.01)
        # 检查退出条件
        if robotic_data['EXIT']:
            print('系统退出')
            break