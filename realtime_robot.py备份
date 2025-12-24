import time

import warnings
import sys
# 忽略urdfpy的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urdfpy")

import numpy as np
import cv2
import torch
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from handmocap.hand_mocap_api import HandMocap
from bodymocap.body_mocap_api import BodyMocap
from handmocap.hand_bbox_detector import HandBboxDetector
from integration.copy_and_paste import integration_copy_paste
from mylib.demo_options import DemoOptions, __filter_bbox_list
from mylib.simu2realrobot import *
from mylib import mocap2infer

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as transforms
from torch_geometric.data import Batch, DataListLoader
from tensorboardX import SummaryWriter
import argparse
import logging
import time
import os
import copy
from datetime import datetime

import dataset
from dataset import Normalize, parse_hand, parse_hand_realtime, parse_body_hand_realtime
from models import model_hand,model_ur3
from models.loss import CollisionLoss, JointLimitLoss, RegLoss, calculate_all_loss
from train import train_epoch
from utils.config import cfg
from utils.util import create_folder

import gym, yumi_gym
import pybullet as p
import numpy as np
import time
import queue as queue_module  # 避免与multiprocessing.queue冲突
import keyboard

import time
from Robotic_Arm.rm_robot_interface import *
import math


def normalize_hand_orientation(hand_xyz):
    wrist = 0
    z_dir = hand_xyz[3] - wrist
    z_dir /= np.linalg.norm(z_dir)
    v1 = hand_xyz[0] - wrist
    v2 = hand_xyz[6] - wrist
    y_dir = np.cross(v1, v2)
    y_dir /= np.linalg.norm(y_dir)
    x_dir = np.cross(y_dir, z_dir)
    x_dir /= np.linalg.norm(x_dir)
    R = np.stack([x_dir, y_dir, z_dir], axis=1)
    aligned = (hand_xyz - wrist) @ R
    return aligned


# 定义进程 1
def hand_capture_process(queue,queue1,queue2,queue_monitor1,stop_event):

    # 初始化手部检测模型：加载预训练的手部姿态估计模型
    # 连接RealSense相机：获取实时视频流
    # 手部检测与姿态估计：使用FrankMocap模型检测手部并估计3D关节点位置
    # 3D骨架可视化
    # 进程间通信：将检测到的手部数据通过队列传递给其他进程
    #循环1000次退出循环
    count = 0
    
    # === 关节点处理 ===
    remove_indices = [0, 7, 11, 15, 19]
    reorder_indices = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3]
    # === 骨架连接关系 ===
    connections = [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9),
        (0, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16)
    ]
    # connections = [
    #     (0, 1), (1, 2), (2, 3),
    #     (0, 4), (4, 5), (5, 6),
    #     (0, 7), (7, 8), (8, 9),
    #     (0, 10), (10, 11), (11, 12),
    #     (0, 13), (13, 14), (14, 15), (15, 16)
    # ]
    connections_body = [(0,2),(2,4),(1,3),(3,5)]
    # === 模型初始化 ===
    args = DemoOptions().parse()
    checkpoint_hand = "./extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
    smpl_dir = "./extra_data/smpl"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    hand_mocap = HandMocap(checkpoint_hand, smpl_dir, device=device, use_smplx=True)
    bbox_detector = HandBboxDetector("third_view", device=device)
    body_mocap = BodyMocap(args.checkpoint_body_smplx, smpl_dir, device=device, use_smplx=True)
    
    # === RealSense 初始化 ===
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        profile = pipeline.start(config)
        print("RealSense相机初始化成功")
    except Exception as e:
        print("无法连接RealSense相机:", e)
#### 绘制 ####
    draw_in_window = 0
    if draw_in_window:
        # === matplotlib 初始化 ===
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        fig.canvas.set_window_title('Hand Capture Process')  # 添加窗口标题
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        ax.view_init(elev=20, azim=5)
        # 添加坐标轴标签
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.ion()
        scat_left = ax.scatter([], [], [], c='b', label="Left",s=5)
        scat_right = ax.scatter([], [], [], c='r', label="Right",s=5)
        lines_left = [ax.plot([], [], [], 'b-')[0] for _ in connections]
        lines_right = [ax.plot([], [], [], 'r-')[0] for _ in connections]
        
        scat_body = ax.scatter([], [], [], c='g', label="Body",s=40)
        lines_body = [ax.plot([], [], [], 'g-')[0] for _ in connections_body]
#### 绘制 ####
    print("启动实时手部识别与 3D 骨架渲染...")
    cap = cv2.VideoCapture(0)  # 打开默认摄像头(设备ID为0)
    # 数据初始化
    joints3d_l = [
        [2.23053899e-02, 1.06128262e-09, 9.62396115e-02],
        [3.19598131e-02, 3.90516105e-03, 1.26066595e-01],
        [4.97228280e-02, 1.94822848e-02, 1.66402832e-01],
        [-1.34232514e-09, 3.73889622e-03, 1.04027525e-01],
        [-2.42368644e-03, 8.80356599e-03, 1.33873671e-01],
        [-1.28910495e-02, 2.57444698e-02, 1.80630386e-01],
        [-2.31615044e-02, 6.77688128e-10, 9.34488773e-02],
        [-2.36971509e-02, -2.46919561e-02, 1.05353259e-01],
        [-1.31559931e-02, -4.02892902e-02, 6.31174967e-02],
        [-3.97054367e-02, -6.32138131e-03, 8.15211609e-02],
        [-4.10398319e-02, -2.38247141e-02, 8.82319063e-02],
        [-2.54071932e-02, -4.54840250e-02, 6.02781698e-02],
        [2.58057639e-02, -1.62365418e-02, 3.77253816e-02],
        [1.84682235e-02, -3.72028612e-02, 5.60834557e-02],
        [6.85350643e-03, -4.27864157e-02, 7.79465064e-02],
        [-2.03244798e-02, -5.20831421e-02, 8.73415545e-02]
    ]
    joints3d_r = [
        [2.20643897e-02, -2.44976484e-10, 9.50904489e-02],
        [2.93233413e-02, 1.57112931e-03, 1.25490069e-01],
        [4.39473838e-02, 1.57856476e-02, 1.68182105e-01],
        [-8.22726332e-10, 3.70741333e-03, 1.02786452e-01],
        [-4.61505726e-03, 7.88622163e-03, 1.32256597e-01],
        [-1.58755854e-02, 2.11675521e-02, 1.80069894e-01],
        [-2.29191463e-02, -3.67799124e-10, 9.22910273e-02],
        [-2.86155604e-02, -2.42707636e-02, 1.03186712e-01],
        [-2.12326739e-02, -3.06558330e-02, 5.93488812e-02],
        [-3.93107943e-02, -6.26177387e-03, 8.04897249e-02],
        [-4.61889058e-02, -2.12232582e-02, 8.90969858e-02],
        [-3.85004729e-02, -4.44807783e-02, 6.27480820e-02],
        [2.54545826e-02, -1.60328317e-02, 3.72054093e-02],
        [1.58985015e-02, -3.71224321e-02, 5.39116226e-02],
        [-6.76039839e-04, -4.15633917e-02, 7.23964795e-02],
        [-2.92139538e-02, -4.05415148e-02, 8.10371637e-02]
    ]
    # 前端繁忙符号 为1时候不再生成数据
    busy_show = 0
    flag2 = 1
    flag3 = 0
    while True:
        # print(1111)
        count += 1
        t0 = time.time()
        # 检查是否需要停止
        if stop_event and stop_event.is_set():
            print("收到停止信好,hand_capture进程退出")
            break
        if count % 20 == 0:
            print('进程1',count/100)
        if count > 10000:
            break
        # if busy_show == 1:
        #     try:
        #         # 使用超时，例如 0.1 秒，避免长时间阻塞
        #         complete_show = queue2.get(timeout=0.1)
        #         if complete_show == 1:
        #             busy_show = 0
        #             print("收到渲染重定向完成信号，继续处理新帧")
        #     except queue_module.Empty:
        #         pass
        #     continue
        
        # 检查相机是否会超时
        try:
            # 设置超时时间为1000毫秒（1秒）
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            wait_duration = time.time() - t0
            if wait_duration > 0.05:  # 如果等待时间超过0.5秒
                print(f"注意: 获取相机帧耗时较长 {wait_duration:.2f} 秒")
        except RuntimeError as e:
            print("相机为空或被占用:", e)
            continue
        img = np.asanyarray(color_frame.get_data())
        # #将本帧图片转为数组 然后调用frankmocap获取3D关键点
        t1 = time.time()
        
        
        # ret, img = cap.read() 
        # 读取本地图片
        # img =  cv2.imread('./download.png')
        # img = cv2.imread('./000067.jpg')
        detect_output = bbox_detector.detect_hand_bbox(img.copy())

        body_pose_list, body_bbox_list, hand_bbox_list, _ = detect_output
        # 根据边界框大小排序，并根据single_person参数决定是否只保留一个目标
        body_bbox_list, hand_bbox_list = __filter_bbox_list(
            body_bbox_list, hand_bbox_list, 1)
        
        # 分别对手部和人体进行姿态回归
        # pred_hand_list = hand_mocap.regress(img, hand_bbox_list, add_margin=True)
        pred_hand_list = hand_mocap.regress(img, hand_bbox_list, add_margin=True)
        pred_body_list = body_mocap.regress(img, body_bbox_list)
        assert len(hand_bbox_list) == len(pred_hand_list)
        
        pred_output_list = pred_hand_list
        hand_detected = 0
        detected = 0
        integral_output_list = integration_copy_paste(
        pred_body_list, pred_hand_list, body_mocap.smpl, img.shape)
        try :
            integral_output = integral_output_list[0]
            l_hand_3D =integral_output['l_pred_hand_joints_3d']
            r_hand_3D =integral_output['r_pred_hand_joints_3d']
            body_3D = integral_output['pred_joints_3d']
            hand_detected = 1 
            body_detected = 1
        except :
            print("no detected") 
            continue
        # if pred_output_list is None:
        #     print('本帧无手部检测结果')
        #     continue
        # else:
        #     try :
        #         pred_output = pred_output_list[0]
        #     except IndexError:
        #         print('本帧无手部检测结果')
        #         continue
        #     if not pred_output:
        #         print('本帧无手部检测结果')
        #         continue
        #     print('本帧有手部检测结果')
        #     hand_detected =1
        # if  pred_body_list is not None:
        #     pred_body = pred_body_list[0]
        #     if pred_body is not None:
        #         print('本帧有身体检测结果')
        #         body_detected = 1
        #     else :
        #         body_detected = 0
        #         body_pose = 0
        #         body_3D = 0
        #         print('本帧无身体检测结果')
        # else :
        #     body_detected = 0
        #     body_pose = 0
        #     body_3D = 0
        #     print('本帧无身体检测结果')

        if hand_detected and body_detected:
            pred_body = integral_output_list[0]
            # print(integral_output_list)
            body_pose,body_3D,l_hand_vec,r_hand_vec = mocap2infer.select_data(pred_body,integral_output_list[0])
            print(f"Action compute time frankmocap: {time.time() - t1:.4f} s")
            detected = 1
            print('检测到结果 进行渲染')
            # if pred_output['left_hand'] is None:
            #     print('没有左手数据')
            #     continue
            # if pred_output['right_hand'] is None:
            #     print('没有右手数据')
            #     continue
            l_hand_3D =integral_output['l_pred_hand_joints_3d']
            r_hand_3D = integral_output['r_pred_hand_joints_3d']
            l_hand_3D_ad = mocap2infer.index_adjust(l_hand_3D, 'left')
            r_hand_3D_ad = mocap2infer.index_adjust(r_hand_3D, 'right')
            #删去初始零点 传出
            l_hand_3D_out = l_hand_3D_ad[1:]
            r_hand_3D_out= r_hand_3D_ad[1:]
###############绘图部分#######################
            if draw_in_window:
                l_hand_3D_ad[:,1]=l_hand_3D_ad[:,1]+0.2
                r_hand_3D_ad[:,1]=r_hand_3D_ad[:,1]-0.2
                # for hand in ['left_hand', 'right_hand']:
                #     if hand in pred_output and pred_output[hand] is not None:
                #         joints2d = pred_output[hand]['pred_joints_img']
                #         for x, y, _ in joints2d:
                #             if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                #                 cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
                scat_left._offsets3d = (l_hand_3D_ad[:, 0],l_hand_3D_ad[:, 1], l_hand_3D_ad[:, 2])
                scat_right._offsets3d = (r_hand_3D_ad[:, 0], r_hand_3D_ad[:, 1], r_hand_3D_ad[:, 2])
                for (i, j), line in zip(connections, lines_left):
                    line.set_data([l_hand_3D_ad[i, 0], l_hand_3D_ad[j, 0]], [l_hand_3D_ad[i, 1], l_hand_3D_ad[j, 1]])
                    line.set_3d_properties([l_hand_3D_ad[i, 2], l_hand_3D_ad[j, 2]])
                for (i, j), line in zip(connections, lines_right):
                    line.set_data([r_hand_3D_ad[i, 0], r_hand_3D_ad[j, 0]], [r_hand_3D_ad[i, 1], r_hand_3D_ad[j, 1]])
                    line.set_3d_properties([r_hand_3D_ad[i, 2], r_hand_3D_ad[j, 2]])
                ##绘制身体数据部分
                joints2d = pred_body['pred_joints_img']
                for x, y, _ in joints2d:
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        cv2.circle(img, (int(x), int(y)), 3, (255, 0, 0), -1)            
                # 从字典中提取选定的关节数据
                selected_joint_names = ['L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']
                # 收集所有关节的坐标数据
                x_coords = []
                y_coords = []
                z_coords = []
                joint_name_all = []
                for joint_name in selected_joint_names:
                    if joint_name in body_3D:
                        joint_data = body_3D[joint_name]
                        x_coords.append(joint_data[0])
                        y_coords.append(joint_data[1])
                        z_coords.append(joint_data[2])
                        joint_name_all.append(joint_name)
                # 更新3D散点图
                scat_body._offsets3d = (x_coords, y_coords, z_coords)
                # 更新连线
                for (i, j), line in zip(connections_body, lines_body):
                    if i < len(x_coords) and j < len(x_coords):
                        x = [x_coords[i], x_coords[j]]
                        y = [y_coords[i], y_coords[j]]
                        z = [z_coords[i], z_coords[j]]
                        line.set_data(x, y)
                        line.set_3d_properties(z)
########绘图部分结束###############
        print('渲染完成')
        ## 通信部分 独立运行时候注释掉         
        if busy_show == 0:
            # print(l_hand_vec,r_hand_vec)
            queue.put(    (     'capture_both_hands', l_hand_3D_out, r_hand_3D_out, 1,body_3D,body_pose,detected,l_hand_vec,r_hand_vec))
            queue_monitor1.put(('capture_both_hands', l_hand_3D_out, r_hand_3D_out, 1,body_3D,body_pose,detected,l_hand_vec,r_hand_vec))
            # robot_queue.put(('capture_both_hands', l_hand_3D_out, r_hand_3D_out, 1,body_3D,body_pose,detected,l_hand_vec,r_hand_vec))
            print('队列长度',queue.qsize(),'监控队列长度',queue_monitor1.qsize())
            busy_show = 1 # 调试时候设为0
            print(f"######frankmocap传出时间: {time.time() - t0:.4f} s")
        elif busy_show == 1:
            # 等待任务完成
            try:
                # 使用超时，例如 0.1 秒，避免长时间阻塞
                complete_show = queue2.get(block=False)  
                if complete_show == 1:
                    busy_show = 0
                    print("#########收到渲染完成信号，继续处理新帧")
            except queue_module.Empty:
                pass 
        cv2.imshow("Camera", img)
        plt.pause(0.001)
        if cv2.waitKey(1) & 0xFF == ord('l'):
            break
        
    pipeline.stop()
    cv2.destroyAllWindows()
    plt.close('all')

def motion_retargeting_process(queue0,queue1,queue2,stop_event,robot_queue):
    '''
    传入数据为'capture_both_hands', joints3d_l, joints3d_r, 1,body_3D,body_pose,body_detected
    '''
    # 解决 torch_geometric 版本兼容性问题
    try:
        from torch_geometric.data import DataEdgeAttr
    except ImportError:
        # 如果当前版本没有 DataEdgeAttr，创建一个兼容类
        class DataEdgeAttr:
            def __init__(self, *args, **kwargs):
                pass
        # 添加到 torch_geometric.data 模块中
        import torch_geometric.data
        torch_geometric.data.DataEdgeAttr = DataEdgeAttr
    
    print('进程2初始化成功')
    count = 0
    parser = argparse.ArgumentParser(description='Inference with trained model')
    parser.add_argument('--cfg', default='configs/inference/yumi.yaml', type=str, help='Path to configuration file')
    args = parser.parse_args()
    # Configurations parse
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print('cfg loaded')
    # print(cfg)

    # Create folder
    create_folder(cfg.OTHERS.LOG)
    create_folder(cfg.OTHERS.SUMMARY)

    # Create logger & tensorboard writer
    log_path = os.path.join(cfg.OTHERS.LOG, "{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now()))
    logger = logging.getLogger('MainLogger')
    logger.setLevel(logging.INFO)
    logger.propagate = True
    # 清除已有 handlers 避免重复
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # 添加新的 FileHandler
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    writer = SummaryWriter(os.path.join(cfg.OTHERS.SUMMARY, "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())))

    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flag1 = 0
    
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
    
    draw_window = 0 #为0不绘图
###### 绘图初始化 ######
    if draw_window :
        # 要绘制关节点位置 所以下面先初始化绘图窗口
        # === matplotlib 初始化 ===
        fig = plt.figure(figsize=(6, 6))
        fig.canvas.set_window_title('Motion Retargeting Process body')  # 添加窗口标题
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.view_init(elev=20, azim=5)
        # 添加坐标轴标签
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        fig2 = plt.figure(figsize=(6, 6))
        ax2 = fig2.add_subplot(111, projection='3d')
        fig2.canvas.set_window_title('hand_retarget')  # 添加窗口标题
        ax2.set_xlim([-0.2, 0.2])
        ax2.set_ylim([-0.2, 0.2])
        ax2.set_zlim([-0.2, 0.2])
        ax2.view_init(elev=-20, azim=270)
        # 添加坐标轴标签
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

        plt.ion()
        connections_body = [(0,1),(1,2),(2,3),(3,4),(4,5),(6,7),(7,8),(8,9),(9,10),(10,11)]
        connections_left =[(0,1),(1,2),(0,3),(3,4),(0,5),(5,6),(0,7),(7,8),(0,10),(10,11),(11,12)]
        connections_right=[(0,1),(1,2),(0,3),(3,4),(0,5),(5,6),(0,7),(7,8),(0,10),(10,11),(11,12)]
        scat_left = ax2.scatter([], [], [], c='b', label="Left")
        scat_right = ax2.scatter([], [], [], c='r', label="Right")
        lines_left = [ax2.plot([], [], [], 'b-')[0] for _ in connections_left]
        lines_right = [ax2.plot([], [], [], 'r-')[0] for _ in connections_right]
        scat_body = ax.scatter([], [], [], c='g', label="Body")
        lines_body = [ax.plot([], [], [], 'g-')[0] for _ in connections_body]
###### 绘图初始化 ######
    
    # Create model
    model = getattr(model_ur3, cfg.MODEL.NAME)().to(device)
    # Load checkpoint
    if cfg.MODEL.CHECKPOINT is not None:
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))
    else:
        #报错 没有模型文件
        assert('没有模型文件')

    while True:
        
        count  +=1
        if count %100 == 0:
            print('进程2运行中')
        if stop_event and stop_event.is_set():
            print("收到停止信好,重定向进程退出")
            break
        task_type, data1, data2, data3, body_3D, body_pose, detected,data4,data5 = queue0.get()
        t1 = time.time()
        if task_type == 'capture_both_hands':
            joints3d_l = data1 #处理 hand_capture 数据
            joints3d_r = data2
            flag1 = data3
        if detected:
            body_3D = body_3D
            body_pose = body_pose
            l_hand_vec = data4
            r_hand_vec = data5
        else:
            #没有检测到
            continue
        #重定向繁忙放入0，等待处理
        queue2.put(0)
        pre_transform = transforms.Compose([Normalize()])
        test_data = parse_body_hand_realtime(joints3d_l,joints3d_r,body_3D,body_pose,l_hand_vec,r_hand_vec)
        test_data = [pre_transform(data) for data in test_data]
        indices = [idx for idx in range(0, len(test_data), cfg.HYPER.BATCH_SIZE)]
        test_loader = [test_data]
        # # Create model
        # model = getattr(model_ur3, cfg.MODEL.NAME)().to(device)
        # # Load checkpoint
        # if cfg.MODEL.CHECKPOINT is not None:
        #     model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))
        # else:
        #     #报错 没有模型文件
        #     assert('没有模型文件')
        
        encode_start_time = time.time()
        model.eval()
        z_all = []
        for batch_idx, data_list in enumerate(test_loader):
                for target_idx, target in enumerate(test_target):
                    # for data in data_list:
                        # # print(data)  # 查看 data 里有哪些属性
                        # # 如果 data 没有 num_nodes，尝试手动赋值
                        # if not hasattr(data, 'num_nodes') or data.num_nodes is None:
                        #     if hasattr(data, 'x') and data.x is not None:
                        #         data.num_nodes = data.x.shape[0]
                        #     elif hasattr(data, 'pos') and data.pos is not None:
                        #         data.num_nodes = data.pos.shape[0]
                        #     else:
                        #         # print(f"Warning: num_nodes is None for data: {data}")
                        #         data.num_nodes = 34  # 保险值（可以根据你的数据调整）
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
        l_hand_ang_all = []
        r_hand_ang_all = []
        l_hand_pos_all = []
        r_hand_pos_all = []
    
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
                if l_hand_ang is not None and r_hand_ang is not None:
                    l_hand_ang_all.append(l_hand_ang)
                    r_hand_ang_all.append(r_hand_ang)
                    l_hand_pos_all.append(l_hand_pos)
                    r_hand_pos_all.append(r_hand_pos)
                if ang is not None:
                    ang_all.append(ang)
                if global_pos is not None:
                    pos_all.append(global_pos)

        if l_hand_ang_all and r_hand_ang_all and ang_all and pos_all:
            l_hand_angle = torch.cat(l_hand_ang_all, dim=0).view(len(test_data),
                                                                    -1).detach().cpu().numpy()
            r_hand_angle = torch.cat(r_hand_ang_all, dim=0).view(len(test_data),
                                                                    -1).detach().cpu().numpy()
            ang = torch.cat(ang_all, dim=0).view(len(test_data),
                                                                    -1).detach().cpu().numpy()
            pos = torch.cat(pos_all, dim=0).view(len(test_data),
                                                                    -1).detach().cpu().numpy()
            l_hand_pos = torch.cat(l_hand_pos_all, dim=0).view(len(test_data),
                                                                    -1).detach().cpu().numpy()
            r_hand_pos = torch.cat(r_hand_pos_all, dim=0).view(len(test_data),
                                                                    -1).detach().cpu().numpy()
            l_hand_angle=l_hand_angle.flatten()
            r_hand_angle=r_hand_angle.flatten()
            l_hand_pos = l_hand_pos.flatten()
            r_hand_pos = r_hand_pos.flatten()
            ang = ang.flatten()
            pos = pos.flatten()
            #将pos每三个作为一行 重组三维坐标
            pos = np.reshape(pos, (-1, 3))
            l_hand_pos = np.reshape(l_hand_pos, (-1, 3))
            r_hand_pos = np.reshape(r_hand_pos, (-1, 3))
            l_hand_pos = np.concatenate([l_hand_pos[0:3],l_hand_pos[4:6],l_hand_pos[7:9],l_hand_pos[10:12],l_hand_pos[13:17]])
            r_hand_pos = np.concatenate([r_hand_pos[0:3],r_hand_pos[4:6],r_hand_pos[7:9],r_hand_pos[10:12],r_hand_pos[13:17]])
        
            # print(l_hand_angle)
            # print(l_hand_angle)
            # print(ang)
            print('重定向完毕')
            # print(l_hand_ang)
            # print(r_hand_ang)
            # print(ang)
            # print('angle check')
            queue1.put(('retargeting_result', l_hand_angle, r_hand_angle,1,ang,detected,pos))  # 放入数据并标记任务类型
            robot_queue.put(('retargeting_result', l_hand_angle,r_hand_angle,1,ang,detected,pos))
            # print('left')
            # print(l_hand_pos)
            # print('right')
            # print(r_hand_pos)
            # print('位置点',l_hand_pos)
############# 更新3D散点图 ############
            if draw_window :
                scat_body._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
                # 更新连线
                for (i, j), line in zip(connections_body, lines_body):
                    if i < len(pos) and j < len(pos):
                        x = [pos[i, 0], pos[j, 0]]
                        y = [pos[i, 1], pos[j, 1]]
                        z = [pos[i, 2], pos[j, 2]]
                        line.set_data(x, y)
                        line.set_3d_properties(z)
                #绘图方便 显示
                l_hand_pos[...,1] += 0.1
                r_hand_pos[...,1] -= 0.1

                scat_left.get_offsets = (l_hand_pos[:,0],l_hand_pos[:,1],l_hand_pos[:,2])
                for (i, j), line in zip(connections_left, lines_left):
                    if i < len(l_hand_pos) and j < len(l_hand_pos):
                        x = [l_hand_pos[i, 0], l_hand_pos[j, 0]]
                        y = [l_hand_pos[i, 1], l_hand_pos[j, 1]]
                        z = [l_hand_pos[i, 2], l_hand_pos[j, 2]]
                        line.set_data(x, y)
                        line.set_3d_properties(z)

                scat_right.get_offsets = (r_hand_pos[:,0],r_hand_pos[:,1],r_hand_pos[:,2])
                for (i, j), line in zip(connections_right, lines_right):
                    if i < len(r_hand_pos) and j < len(r_hand_pos):
                        x = [r_hand_pos[i, 0], r_hand_pos[j, 0]]
                        y = [r_hand_pos[i, 1], r_hand_pos[j, 1]]
                        z = [r_hand_pos[i, 2], r_hand_pos[j, 2]]
                        line.set_data(x, y)
                        line.set_3d_properties(z)
# ############ 结束绘制 ###############
            # 刷新图形
            plt.draw()
        print(f"Motion retargeting time: {time.time() - t1:.4f} s")

def Pybullet_control_process(queue,queue1,queue2,stop_event):
    
    import warnings
    # 忽略碰撞警告
    warnings.filterwarnings("ignore", message=".*Collision Occurred in Joint.*")
    # l_joint_angle_np = [-1, -0.821, -0.726, -0.042, 0.905, 0.116, 0.179]
    # r_joint_angle_np = [1, -0.811, 0.663, 0.095, -1, 0.158, -0.158]
    l_hand_angle = np.zeros(18)
    r_hand_angle = np.zeros(18)
    l_joint_angle_np = [-1, -0.821, -0.726, -0.042, 0.905, 0.116]
    r_joint_angle_np = [1, -0.811, 0.663, 0.095, -1, 0.158]
    body_ang = np.zeros(12)
    env = gym.make('yumi-v0')
    observation = env.reset()

    camera_distance = 2
    camera_yaw = 140
    camera_pitch = -15
    camera_target_position = [0, 0.2, 0.3]

    frame_id = 0  # 新增帧计数器
    flag2 = 0
    temp_data1 = None
    temp_data2 = None
    temp_data3 = None
    temp_data4 = None
    temp_task = None
    mocap_data = 0
    print("已初始化仿真平台")
    while True:
        if stop_event.is_set():
            print("终止虚拟仿真平台")
            break
        try :
            task_type, data1, data2, data3,data4,detected,_= queue1.get(block=False)
            temp_task,temp_data1,temp_data2,temp_data3,temp_data4,detected= task_type, data1, data2, data3,data4,detected
            mocap_data = 1
        except :
            if temp_data1 is not None:
                task_type, data1, data2, data3,data4,detected = temp_task, temp_data1, temp_data2, temp_data3,temp_data4,True
                mocap_data = 0
            else : continue
        t2=time.time()
        if mocap_data:
            queue2.put(0)
        if detected:
            body_ang = data4
            # print('虚拟仿真')
        else :
            continue
        #读取数据
        if task_type == 'retargeting_result':
            l_hand_angle, r_hand_angle, flag2 = data1, data2, data3
        # print('身体角度',body_ang,'左手角度',l_hand_angle,'右手角度',r_hand_angle)
        # remove zeros
        # print(l_hand_angle.shape)
        # print(r_hand_angle.shape)
        l_hand_angle = np.concatenate([l_hand_angle[0:3],l_hand_angle[4:6],l_hand_angle[7:9],l_hand_angle[10:12],l_hand_angle[13:17]])
        r_hand_angle = np.concatenate([r_hand_angle[0:3],r_hand_angle[4:6],r_hand_angle[7:9],r_hand_angle[10:12],r_hand_angle[13:17]])
        # print(l_hand_angle.shape)
        # print(r_hand_angle.shape)
        # print(body_ang.shape)
        # 每两帧才渲染一次
        env.render()
        p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                     cameraYaw=camera_yaw,
                                     cameraPitch=camera_pitch,
                                     cameraTargetPosition=camera_target_position)
        action = np.concatenate([
            body_ang,
            l_hand_angle,
            r_hand_angle
        ])
        # print('#######组合后角度',action)
        # print(f"action: {action}")
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
                elif k == ord('j'):
                    camera_target_position[0] -= 0.3
                elif k == ord('l'):
                    camera_target_position[0] += 0.3
                elif k == ord('i'):
                    camera_target_position[1] += 0.3
                elif k == ord('k'):
                    camera_target_position[1] -= 0.3
                elif k == ord('u'):
                    camera_target_position[2] += 0.3
                elif k == ord('o'):
                    camera_target_position[2] -= 0.3
        observation, reward, done, info = env.step(action)
        time.sleep(0.002)
        flag2 = 0
        frame_id += 1  # 更新帧编号
        if mocap_data:
            queue2.put((1))  # 放入数据并标记任务类型
            print(f"Pybullet render time: {time.time() - t2:.4f} s")

def control_robot(queue,queue1,queue2,stop_event,robot_queue):
    l_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    l_handle = l_arm.rm_create_robot_arm("192.168.10.19", 8080)
    r_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    r_handle = r_arm.rm_create_robot_arm("192.168.10.18", 8080)
    print("机械臂ID,", l_handle.id)
    min = [-178.0, -130.0, -135.0, -180.0, -128.0, -360.0]
    max = [178.0, 130.0, 135.0, 180.0, 128.0, 360.0]
    # L_robot_arm_angle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # R_robot_arm_angle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    L_robot_arm_angle = [0.0, -90, 0, 0.0, 0.0, 0.0]
    R_robot_arm_angle = [0.0, -90, 0, 0.0, 0.0, 0.0]
    stop = False
    # 初始化位置
    l_arm.rm_movej(L_robot_arm_angle, 50, 0, 0, 1)
    r_arm.rm_movej(R_robot_arm_angle, 50, 0, 0, 1)
    temp_data1 = None
    temp_data2 = None
    temp_data3 = None
    temp_data4 = None
    temp_task = None
    mocap_data = 0
    while not stop:
        if stop_event.is_set():
                print("终止机器人平台")
                break
        try :
            task_type, data1, data2, data3,data4,detected,_= robot_queue.get(block=False)
            temp_task,temp_data1,temp_data2,temp_data3,temp_data4,detected= task_type, data1, data2, data3,data4,detected
            mocap_data = 1
            print("机器人端收到数据")
        except :
            if temp_data1 is not None:
                task_type, data1, data2, data3,data4,detected = temp_task, temp_data1, temp_data2, temp_data3,temp_data4,True
                mocap_data = 0
            else : 
                # print('机器人没有数据')
                continue
        t2=time.time()
        # if mocap_data:
        #     queue2.put(0)
        if detected:
            body_ang = data4
            print('虚拟仿真')
        else :
            continue
        #读取数据
        if task_type == 'retargeting_result':
            l_hand_angle, r_hand_angle, flag2 = data1, data2, data3
        # print('身体角度',body_ang,'左手角度',l_hand_angle,'右手角度',r_hand_angle)
        # remove zeros
        # print(l_hand_angle.shape)
        # print(r_hand_angle.shape)
        l_hand_angle = np.concatenate([l_hand_angle[0:3],l_hand_angle[4:6],l_hand_angle[7:9],l_hand_angle[10:12],l_hand_angle[13:17]])
        r_hand_angle = np.concatenate([r_hand_angle[0:3],r_hand_angle[4:6],r_hand_angle[7:9],r_hand_angle[10:12],r_hand_angle[13:17]])
    
    ##############
        #前一半左后一半右
        l_joint_angle = body_ang[0:6]
        r_joint_angle = body_ang[6:12]
        l_joint_angle_np = np.array(l_joint_angle)
        r_joint_angle_np = np.array(r_joint_angle)
        L_robot_arm_angle = trans2realworld1(l_joint_angle_np,min,max)
        R_robot_arm_angle = trans2realworld(r_joint_angle_np,min,max)
        R_robot_hand_angle = trans2realworld_hand(r_hand_angle)
        print('左手手臂',L_robot_arm_angle)
        print('右手手臂',R_robot_arm_angle)
        # print('左臂命令',l_arm.rm_movej_canfd(L_robot_arm_angle, False, 0, 1, 50),'原始角度',l_joint_angle_np)
        # time.sleep(0.10)
        # print('左手命令',l_arm.rm_movej(L_robot_arm_angle, 40, 0, 0, 1))
        print('右臂命令',r_arm.rm_movej_canfd(R_robot_arm_angle, False, 0, 1, 50),'原始角度',r_joint_angle_np)
        time.sleep(0.5)
        # print('右手命令',r_arm.rm_set_hand_follow_angle([0,100,200,300,400,500], True))
        if keyboard.is_pressed('esc'):
            stop = True
            stop_event.set()
            break
        if stop: 
            stop_event.set()
            break
    
    # 断开所有连接
    print("断开所有连接")
    l_arm.rm_movej([0.0, -90, 0, 0.0, 0.0, 0.0], 50, 0, 0, 1)
    # l_arm.rm_set_hand_follow_angle([0,0,0,0,0,0], True)
    r_arm.rm_movej([0.0, -90, 0, 0.0, 0.0, 0.0], 50, 0, 0, 1)
    RoboticArm.rm_destroy()
# 创建一个测试函数来验证队列通信
def test_queue_output(queue_monitor,stop_event):
    """
    测试队列通信的函数，同时承担消化队列数据的责任
    """
    print("测试进程启动，开始监控队列...")
    processed_count = 0
    stop = 0
    while stop < 50:
        if stop_event and stop_event.is_set():
            print("收到停止信号，测试进程退出")
            break
        try:
                data = queue_monitor.get()
                processed_count += 1
                # 解析你放入队列的元组 ('capture_both_hands', joints3d_l, joints3d_r, 1)
                task_type, joints_l, joints_r, flag = data
                print(f"从队列接收到数据:")
                print(f"  任务类型: {task_type}")
                # print(f"  左手关节数量: {len(joints_l) if joints_l is not None else 0}")
                # print(f"  右手关节数量: {len(joints_r) if joints_r is not None else 0}")
                # print(f"  标志位: {flag}")
                stop =0
        except Exception as e:
                print(f"队列为空(已处理 {processed_count} 条数据)")
                stop += 1
    print("测试进程已终止")


def keyboard_monitor_process(stop_event):
    """
    监控键盘输入的进程,按ESC或'q'键终止所有进程
    """
    print("键盘监控进程已启动,按ESC或'q'键终止所有进程")
    while not stop_event.is_set():
        try:
            if keyboard.is_pressed('esc') :
                print("检测到终止按键，正在停止所有进程...")
                stop_event.set()
                break
            time.sleep(0.1)  # 避免过度占用CPU
        except Exception as e:
            print(f"键盘监控出错: {e}")
    print("键盘监控进程已终止")

if __name__ == '__main__':
    from multiprocessing import Process, Queue, Event

    queue = Queue()
    queue1 = Queue()
    queue2 = Queue()

    queue_monitor1 = Queue()
    robot_queue = Queue()
    # 创建停止事件
    stop_event = Event()
    # 创建并启动所有进程
    # test_process = Process(target=test_queue_output, args=(queue_monitor1, stop_event))
    p1 = Process(target=hand_capture_process, args=(queue, queue1, queue2, queue_monitor1,stop_event))
    p2 = Process(target=motion_retargeting_process, args=(queue, queue1, queue2,stop_event,robot_queue))
    keyboard_process = Process(target=keyboard_monitor_process, args=(stop_event,))
    p3 = Process(target=Pybullet_control_process, args=(queue, queue1, queue2,stop_event))
    # p1 = Process(target=hand_capture_process, args=(queue, queue1, queue2,queue_monitor1))
    p4 = Process(target=control_robot, args=(queue, queue1, queue2,stop_event,robot_queue))
    

    # 启动所有进程
    # test_process.start()
    p1.start()
    keyboard_process.start()
    p2.start()
    p3.start()
    p4.start()


