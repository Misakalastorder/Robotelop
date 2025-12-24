import sys
import time

import xrobotoolkit_sdk as xrt
from multiprocessing import Manager, Process

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    filename='test_100.h5'
    frame_num = 100
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
    try:
        print("Initializing SDK...")
        xrt.init()
        print("SDK Initialized successfully.")

        print("\n--- Testing all functions for 10 iterations ---")
        for i in range(frame_num):
            print(f"\n--- Iteration {i+1} ---")
            left_pose = xrt.get_left_controller_pose()
            right_pose = xrt.get_right_controller_pose()
            headset_pose = xrt.get_headset_pose()

            # print(f"Left Controller Pose: {left_pose}")
            # print(f"Right Controller Pose: {right_pose}")
            # print(f"Headset Pose: {headset_pose}")
            # print('********上面为三姿态')
            # Check if body tracking data is available
            if xrt.is_body_data_available():
                # Get body joint poses (24 joints, 7 values each: x,y,z,qx,qy,qz,qw)
                body_poses = xrt.get_body_joints_pose()
                # print(f"Body joints pose data: {body_poses}")
                robotic_data['body'] = body_poses
                # Get body data timestamp
                body_timestamp = xrt.get_body_timestamp_ns()
                # print(f"Body data timestamp: {body_timestamp}")
                
                # Example: Get specific joint data (Head joint is index 15)
                head_pose = body_poses[15]  # Head joint
                x, y, z, qx, qy, qz, qw = head_pose
                robotic_data['head'] = [x, y, z]
                # print(f"Head pose: Position({x:.3f}, {y:.3f}, {z:.3f}) Rotation({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")
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
                    # 保存时间戳
                    ts_dataset = h5f['timestamp']
                    ts_dataset.resize((ts_dataset.shape[0] + 1,))
                    ts_dataset[-1] = body_timestamp
            else:
                print("Body tracking data not available. Make sure:")
                print("1. PICO headset is connected")
                print("2. Body tracking is enabled in the control panel")
                print("3. At least two Pico Swift devices are connected and calibrated")

            time.sleep(0.5)  # Wait for 0.5 seconds before the next iteration
        robotic_data['EXIT'] = True
        print("\nAll iterations complete.")
        if save:
            h5f.close()
    except RuntimeError as e:
        print(f"Runtime Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        print("\nClosing SDK...")
        xrt.close()
        print("SDK closed.")
        print("Test finished.")

def show_data(robotic_data):
    '''
    显示数据，定时刷新
    '''
    # 创建3D图形窗口
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴范围和标签
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Real-time Body Tracking Data')
    
    while not robotic_data['EXIT']:
        # 清除之前的点
        ax.cla()
        # 重新设置坐标轴范围和标签
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Real-time Body Tracking Data')
        
        # 获取身体位置数据
        body_data = robotic_data.get('body')
        
        if body_data is not None:
            # 提取所有关节的位置坐标(x, y, z)
            positions = np.array([[joint[0], joint[1], joint[2]] for joint in body_data])
            
            # 绘制所有关节
            xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
            ax.scatter(xs, ys, zs, c='r', marker='o')
            
            # 为每个关节点添加标签
            for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
                ax.text(x, y, z, f'{joint_names[i]}', fontsize=8)
                
        # 刷新图形
        plt.pause(0.1)
        
    plt.close(fig)



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
    get_data_process = Process(target=run_data_tests, args=(robotic_data,))
    get_data_process.daemon = True
    get_data_process.start()

    show_data_process = Process(target=show_data, args=(robotic_data,))
    show_data_process.daemon = True
    show_data_process.start()
    while True:
        time.sleep(0.01)
        # 检查退出条件
        if robotic_data['EXIT']:
            print('系统退出')
            break