import os
import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import pybullet_data
import numpy as np
data_path = pybullet_data.getDataPath()
# robot_path = os.path.join(data_path, 'DACL_description', 'urdf' ,'DACL_description_test1.urdf')
# D:\2025\crp\yushu_robot\yushu_robot\yumi_gym\envs
# D:\2025\crp\yushu_robot\yushu_robot\h1_description
# D:\2025\crp\yushu_robot\h1_description\urdf\h1_with_hand.urdf  data\target\ur3\robot(ur3).urdf
# C:\Users\30687\Desktop\ur3_robot_train_pico\ur3_robot_train_pico\data\target\ur3\robot(ur3).urdf
# D:\2025\Roboteleop\Robotelop\configs
robot_path = os.path.join('D:\\2025\\Roboteleop\\Robotelop\\data\\target\\ur3\\robot(ur3).urdf')
class YumiEnv(gym.Env):
    """docstring for YumiEnv"""
    def __init__(self):
        super(YumiEnv, self).__init__()
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[0,0,0.06])
        self.step_counter = 0
        self.joints = [
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

        # 'yumi_link_7_l_joint',
        # 'link1',
        # 'link11',
        # 'link2',
        # 'link22',
        # 'link3',
        # 'link33',
        # 'link4',
        # 'link44',
        # 'link5',
        # 'link51',
        # 'link52',
        # 'link53',

        # 'yumi_link_7_r_joint',
        # 'Link1',
        # 'Link11',
        # 'Link2',
        # 'Link22',
        # 'Link3',
        # 'Link33',
        # 'Link4',
        # 'Link44',
        # 'Link5',
        # 'Link51',
        # 'Link52',
        # 'Link53',

            ]
        self.action_space = spaces.Box(np.array([-1]*len(self.joints)), np.array([1]*len(self.joints)))
        self.observation_space = spaces.Box(np.array([-1]*len(self.joints)), np.array([1]*len(self.joints)))

    def step(self, action, custom_reward=None):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        # print(self.joint2Index["yumi_joint_1_l"])
        # print(len(action))
        # print(self.joints)
        p.setJointMotorControlArray(self.yumiUid, [self.joint2Index[joint] for joint in self.joints], p.POSITION_CONTROL, action)

        p.stepSimulation()

        jointStates = {}
        for joint in self.joints:
            jointStates[joint] = p.getJointState(self.yumiUid, self.joint2Index[joint]) + p.getLinkState(self.yumiUid, self.joint2Index[joint])
        # recover color
        for joint, index in self.joint2Index.items():
            if joint in self.jointColor and joint != 'world_joint':
                p.changeVisualShape(self.yumiUid, index, rgbaColor=self.jointColor[joint])
        # check collision
        collision = False
        # for joint in self.joints:
        #     if len(p.getContactPoints(bodyA=self.yumiUid, linkIndexA=self.joint2Index[joint])) > 0:
        #         collision = True
        #         for contact in p.getContactPoints(bodyA=self.yumiUid, linkIndexA=self.joint2Index[joint]):
        #             print("Collision Occurred in Joint {} & Joint {}!!!".format(contact[3], contact[4]))
        #             p.changeVisualShape(self.yumiUid, contact[3], rgbaColor=[1,0,0,1])
        #             p.changeVisualShape(self.yumiUid, contact[4], rgbaColor=[1,0,0,1])
        
        self.step_counter += 1

        if custom_reward is None:
            # default reward
            reward = 0
            done = False
        else:
            # custom reward
            reward, done = custom_reward(jointStates=jointStates, collision=collision, step_counter=self.step_counter)

        info = {'collision': collision}
        observation = [jointStates[joint][0] for joint in self.joints]
        return observation, reward, done, info

    def reset(self):
        k=0
        p.resetSimulation()
        self.step_counter = 0
        self.yumiUid = p.loadURDF(robot_path, [0,0,0.05],useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        floor = p.loadURDF(pybullet_data.getDataPath() + '/plane.urdf', [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
        # self.tableUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(),
        #     "table/table.urdf"), basePosition=[0,0,-0.65])
        p.setGravity(0,0,-10)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(1./240.)
        self.joint2Index = {} # jointIndex map to jointName
        for i in range(p.getNumJoints(self.yumiUid)):
            self.joint2Index[p.getJointInfo(self.yumiUid, i)[1].decode('utf-8')] = i
        # print(self.joint2Index)
        self.jointColor = {} # jointName map to jointColor

        # print(p.getVisualShapeData(self.yumiUid))
        for data in p.getVisualShapeData(self.yumiUid):
            k=k+1
            # print(p.getJointInfo(self.yumiUid, 0)[1].decode('utf-8'))
            if(k>=2):
                self.jointColor[p.getJointInfo(self.yumiUid, data[1])[1].decode('utf-8')] = data[7]

        # recover color
        for joint, index in self.joint2Index.items():
            if joint in self.jointColor and joint != 'world_joint':
                p.changeVisualShape(self.yumiUid, index, rgbaColor=self.jointColor[joint])


    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.0,0.0,0.05],
                                                          distance=2.2,
                                                          yaw=90,
                                                          pitch=0,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960)/720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960,4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()