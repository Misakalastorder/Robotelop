

# # 创建旋转对象
# rotation = R.from_quat(quaternion)

# # 定义初始X轴向量
# x_axis = np.array([0, 0, -1])
# # 应用旋转得到掌心向量
# palm_vector = rotation.apply(x_axis)

#生成数量和 target_matrix一样 每个数据为(0,0,-1)的张量
# target_raw_vector = torch.tensor([0, 0, -1], dtype=target_matrix.dtype, device=target_matrix.device).repeat(target_matrix.size(0), 1)
# #逐个相乘
# target_vec = torch.bmm(target_matrix, target_raw_vector.view(-1, 3, 1)).view(-1, 3)

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

# 创建一个测试四元数 (qx, qy, qz, qw)
quaternion = np.array([0.1, 0.2, 0.3, 0.9])
quaternion = quaternion / np.linalg.norm(quaternion)  # 归一化

# 创建一个测试向量
vector = np.array([0, 0, -1])

# 方法1: 使用scipy的Rotation对象
rotation = R.from_quat(quaternion)
result1 = rotation.apply(vector)

# 方法2: 使用旋转矩阵和矩阵乘法
rotation_matrix = rotation.as_matrix()
result2 = np.dot(rotation_matrix, vector)

# 比较结果
print("四元数:", quaternion)
print("原始向量:", vector)
print("方法1 (Rotation.apply):", result1)
print("方法2 (矩阵乘法):", result2)
print("结果是否一致:", np.allclose(result1, result2))
