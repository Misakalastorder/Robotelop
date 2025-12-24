import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.conversions import quaternion_to_rotation_matrix

"""
Calculate All Loss
"""


def calculate_all_loss(data_list, target_list, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion,
                       fin_criterion, reg_criterion,el_criterion,
                       z, target_ang, target_pos, target_rot, target_global_pos, l_hand_pos, r_hand_pos, all_losses=[],
                       ee_losses=[], vec_losses=[], col_losses=[], lim_losses=[], ori_losses=[], fin_losses=[],
                       reg_losses=[],el_losses=[]):
    # end effector loss
    if ee_criterion:
        ee_loss = calculate_ee_loss(data_list, target_list, target_pos, ee_criterion) * 1000 # 1000
        # ee_loss = calculate_ee_loss(data_list, target_list, target_global_pos, ee_criterion) * 1000 # 1000
        ee_losses.append(ee_loss.item())
    else:
        ee_loss = 0
        ee_losses.append(0)

    # elbow loss
    if el_criterion:
        el_loss = calculate_el_loss(data_list, target_list, target_pos, el_criterion) * 1000*0  # 1000
        el_losses.append(el_loss.item())
    else:
        el_loss = 0
        el_losses.append(0)

    # vector loss
    if vec_criterion:
        vec_loss = calculate_vec_loss(data_list, target_list, target_pos, vec_criterion) * 100  # 100
        vec_losses.append(vec_loss.item())
    else:
        vec_loss = 0
        vec_losses.append(0)

    # collision loss
    if col_criterion:
        col_loss = col_criterion(target_global_pos.view(len(target_list), -1, 3), target_list[0].edge_index,
                                 target_rot.view(len(target_list), -1, 9), target_list[0].ee_mask) * 1000
        col_losses.append(col_loss.item())
    else:
        col_loss = 0
        col_losses.append(0)

    # joint limit loss
    if lim_criterion:
        lim_loss = calculate_lim_loss(target_list, target_ang, lim_criterion) * 10000
        lim_losses.append(lim_loss.item())
    else:
        lim_loss = 0
        lim_losses.append(0)

    # # end effector orientation loss
    if ori_criterion:
        # ori_loss = calculate_ori_loss(data_list, target_list, target_rot, ori_criterion) * 100  # 100
        ori_loss = calculate_ori_loss6(data_list, target_list, target_rot, ori_criterion, target_pos, target_ang)*100
        # ori_loss = calculate_vec_loss(data_list, target_list, target_pos, vec_criterion)*100
        ori_losses.append(ori_loss.item())
    else:
        ori_loss = 0
        ori_losses.append(0)
    # end effector orientation loss
    # if ori_criterion:
    #     ori_loss = calculate_ori_loss1(data_list, target_list, l_hand_pos, r_hand_pos, ori_criterion) * 100  # 100
    #     ori_losses.append(ori_loss.item())
    # else:
    #     ori_loss = 0
    #     ori_losses.append(0)

    # finger similarity loss
    if fin_criterion:
        fin_loss = calculate_fin_loss(data_list, target_list, l_hand_pos, r_hand_pos, fin_criterion) * 100
        fin_losses.append(fin_loss.item())
    else:
        fin_loss = 0
        fin_losses.append(0)

    # regularization loss
    if reg_criterion:
        reg_loss = reg_criterion(z.view(len(target_list), -1, 64))
        reg_losses.append(reg_loss.item())
    else:
        reg_loss = 0
        reg_losses.append(0)

    # total loss
    # loss = ee_loss + vec_loss + col_loss + lim_loss + ori_loss + fin_loss + reg_loss+el_loss
    loss = ee_loss + vec_loss + col_loss + lim_loss + ori_loss + fin_loss + reg_loss+el_loss
    all_losses.append(loss.item())

    return loss


"""
Calculate End Effector Loss
"""

def calculate_ee_loss(data_list, target_list, target_pos, ee_criterion):
    # print("data_list:",data_list)
    # print("target_list:",target_list)
    # print(target_pos)
    # print("target_pos.size():",target_pos.size())
    target_mask = torch.cat([data.ee_mask for data in target_list]).to(target_pos.device)
    # print("target_mask.size():",target_mask)
    source_mask = torch.cat([data.ee_mask for data in data_list]).to(target_pos.device)
    # print("source_mask.size():",source_mask)
    target_ee = torch.masked_select(target_pos, target_mask).view(-1, 3)
    # print("target_ee.size():",target_ee.size())
    source_ee = torch.masked_select(torch.cat([data.pos for data in data_list]).to(target_pos.device),
                                    source_mask).view(-1, 3)
    # print((torch.cat([data.pos for data in data_list]).to(target_pos.device)).size())
    # print("source_ee.size():",source_ee.size())

    # normalize
    target_root_dist = torch.cat([data.root_dist for data in target_list]).to(target_pos.device)
    source_root_dist = torch.cat([data.root_dist for data in data_list]).to(target_pos.device)
    target_ee = target_ee / torch.masked_select(target_root_dist, target_mask).unsqueeze(1)
    source_ee = source_ee / torch.masked_select(source_root_dist, source_mask).unsqueeze(1)
    # print(target_ee.shape)
    # print(source_ee.shape)
    ee_loss = ee_criterion(target_ee, source_ee)
    return ee_loss

"""
elbow loss
"""
def calculate_el_loss(data_list, target_list, target_pos, ee_criterion):
    # print("data_list:",data_list)
    # print("target_list:",target_list)
    # print(target_pos)
    # print("target_pos.size():",target_pos.size())
    target_mask = torch.cat([data.el_mask for data in target_list]).to(target_pos.device)
    # print("target_mask.size():",target_mask)
    source_mask = torch.cat([data.el_mask for data in data_list]).to(target_pos.device)
    # print("source_mask.size():",source_mask)
    target_ee = torch.masked_select(target_pos, target_mask).view(-1, 3)
    # print("target_ee.size():",target_ee.size())
    source_ee = torch.masked_select(torch.cat([data.pos for data in data_list]).to(target_pos.device),
                                    source_mask).view(-1, 3)
    # print((torch.cat([data.pos for data in data_list]).to(target_pos.device)).size())
    # print("source_ee.size():",source_ee.size())

    # normalize
    target_root_dist = torch.cat([data.root_dist for data in target_list]).to(target_pos.device)
    source_root_dist = torch.cat([data.root_dist for data in data_list]).to(target_pos.device)
    target_ee = target_ee / torch.masked_select(target_root_dist, target_mask).unsqueeze(1)
    source_ee = source_ee / torch.masked_select(source_root_dist, source_mask).unsqueeze(1)
    # print(target_ee.shape)
    # print(source_ee.shape)
    el_loss = ee_criterion(target_ee, source_ee)
    return el_loss


"""
Calculate Vector Loss
"""
def calculate_vec_loss(data_list, target_list, target_pos, vec_criterion):
    target_sh_mask = torch.cat([data.sh_mask for data in target_list]).to(target_pos.device)
    target_el_mask = torch.cat([data.el_mask for data in target_list]).to(target_pos.device)
    target_ee_mask = torch.cat([data.ee_mask for data in target_list]).to(target_pos.device)
    source_sh_mask = torch.cat([data.sh_mask for data in data_list]).to(target_pos.device)
    source_el_mask = torch.cat([data.el_mask for data in data_list]).to(target_pos.device)
    source_ee_mask = torch.cat([data.ee_mask for data in data_list]).to(target_pos.device)
    target_sh = torch.masked_select(target_pos, target_sh_mask).view(-1, 3)
    target_el = torch.masked_select(target_pos, target_el_mask).view(-1, 3)
    target_ee = torch.masked_select(target_pos, target_ee_mask).view(-1, 3)
    source_sh = torch.masked_select(torch.cat([data.pos for data in data_list]).to(target_pos.device),
                                    source_sh_mask).view(-1, 3)
    source_el = torch.masked_select(torch.cat([data.pos for data in data_list]).to(target_pos.device),
                                    source_el_mask).view(-1, 3)
    source_ee = torch.masked_select(torch.cat([data.pos for data in data_list]).to(target_pos.device),
                                    source_ee_mask).view(-1, 3)
    # print(target_sh.shape, target_el.shape, target_ee.shape, source_sh.shape, source_el.shape, source_ee.shape)
    target_vector1 = target_el - target_sh
    target_vector2 = target_ee - target_el
    source_vector1 = source_el - source_sh
    source_vector2 = source_ee - source_el
    # print(target_vector1.shape, target_vector2.shape, source_vector1.shape, source_vector2.shape, (target_vector1*source_vector1).sum(-1).shape)
    # normalize
    target_shoulder_dist = torch.cat([data.shoulder_dist for data in target_list]).to(target_pos.device)
    target_elbow_dist = torch.cat([data.elbow_dist for data in target_list]).to(target_pos.device) / 2
    source_shoulder_dist = torch.cat([data.shoulder_dist for data in data_list]).to(target_pos.device)
    source_elbow_dist = torch.cat([data.elbow_dist for data in data_list]).to(target_pos.device) / 2
    normalize_target_vector1 = target_vector1 / torch.masked_select(target_shoulder_dist, target_el_mask).unsqueeze(1)
    normalize_source_vector1 = source_vector1 / torch.masked_select(source_shoulder_dist, source_el_mask).unsqueeze(1)
    vector1_loss = vec_criterion(normalize_target_vector1, normalize_source_vector1)
    #计算两组向量的夹角作为loss
    # 使用 CosineEmbeddingLoss 计算向量1的角度损失
    cosine_embedding_loss = nn.CosineEmbeddingLoss(reduction='mean')
    targets = torch.ones(normalize_target_vector1.size(0), device=normalize_target_vector1.device)
    vector1_loss_ang = cosine_embedding_loss(normalize_target_vector1, normalize_source_vector1, targets)
    print("vector1_loss_ang:", vector1_loss_ang)

    normalize_target_vector2 = target_vector2 / torch.masked_select(target_elbow_dist, target_ee_mask).unsqueeze(1)
    normalize_source_vector2 = source_vector2 / torch.masked_select(source_elbow_dist, source_ee_mask).unsqueeze(1)
    vector2_loss = vec_criterion(normalize_target_vector2, normalize_source_vector2)
    # 使用 CosineEmbeddingLoss 计算向量2的角度损失
    targets = torch.ones(normalize_target_vector2.size(0), device=normalize_target_vector2.device)
    vector2_loss_ang = cosine_embedding_loss(normalize_target_vector2, normalize_source_vector2, targets)
    print("vector2_loss_ang:", vector2_loss_ang)
    
    vec_loss = vector2_loss  # (vector1_loss + vector2_loss)*100
    vec_loss = (vector1_loss + vector2_loss)/2
    return vec_loss

def calculate_vec_loss2(data_list, target_list, target_pos, vec_criterion):
    target_sh_mask = torch.cat([data.sh_mask for data in target_list]).to(target_pos.device)
    target_el_mask = torch.cat([data.el_mask for data in target_list]).to(target_pos.device)
    target_ee_mask = torch.cat([data.ee_mask for data in target_list]).to(target_pos.device)
    source_sh_mask = torch.cat([data.sh_mask for data in data_list]).to(target_pos.device)
    source_el_mask = torch.cat([data.el_mask for data in data_list]).to(target_pos.device)
    source_ee_mask = torch.cat([data.ee_mask for data in data_list]).to(target_pos.device)
    target_sh = torch.masked_select(target_pos, target_sh_mask).view(-1, 3)
    target_el = torch.masked_select(target_pos, target_el_mask).view(-1, 3)
    target_ee = torch.masked_select(target_pos, target_ee_mask).view(-1, 3)
    source_sh = torch.masked_select(torch.cat([data.pos for data in data_list]).to(target_pos.device),
                                    source_sh_mask).view(-1, 3)
    source_el = torch.masked_select(torch.cat([data.pos for data in data_list]).to(target_pos.device),
                                    source_el_mask).view(-1, 3)
    source_ee = torch.masked_select(torch.cat([data.pos for data in data_list]).to(target_pos.device),
                                    source_ee_mask).view(-1, 3)
    # print(target_sh.shape, target_el.shape, target_ee.shape, source_sh.shape, source_el.shape, source_ee.shape)
    target_vector1 = target_el - target_sh
    target_vector2 = target_ee - target_el
    source_vector1 = source_el - source_sh
    source_vector2 = source_ee - source_el
    # print(target_vector1.shape, target_vector2.shape, source_vector1.shape, source_vector2.shape, (target_vector1*source_vector1).sum(-1).shape)
    # 使用余弦相似度计算损失
    # 余弦相似度 = (A·B) / (||A|| * ||B||)
    # 损失 = 1 - 余弦相似度 (越接近0越好)
    
    # 计算向量1的余弦相似度损失
    cos_sim1 = torch.cosine_similarity(target_vector1, source_vector1, dim=1)
    vector1_loss = torch.mean(1 - cos_sim1)
    
    # 计算向量2的余弦相似度损失
    cos_sim2 = torch.cosine_similarity(target_vector2, source_vector2, dim=1)
    vector2_loss = torch.mean(1 - cos_sim2)
    
    # 总损失为两个向量损失的平均
    vec_loss = (vector1_loss + vector2_loss) / 2 *10
    return vec_loss
"""
Calculate Joint Limit Loss
"""


def calculate_lim_loss(target_list, target_ang, lim_criterion):
    target_lower = torch.cat([data.lower for data in target_list]).to(target_ang.device)
    target_upper = torch.cat([data.upper for data in target_list]).to(target_ang.device)
    lim_loss = lim_criterion(target_ang, target_lower, target_upper)
    return lim_loss


"""
Calculate Orientation Loss
"""


# # 该函数计算旋转方向的损失（ori_loss），步骤如下：
# # 提取掩码：从target_list和data_list中分别提取ee_mask作为target_mask和source_mask；
# # 处理旋转表示：
# # 将输入的target_rot展平为形状(-1, 9)；
# # 将data_list中的四元数q拼接并转换为旋转矩阵，再展平为(-1, 9)；
# # 应用掩码：使用masked_select根据掩码提取有效旋转信息；
# # 计算损失：用ori_criterion比较目标与源旋转，返回结果。
# # # ori_criterion使用的是nn.MSELoss()
# def calculate_ori_loss(data_list, target_list, target_rot, ori_criterion):
#     # 从 target_list 中提取每个元素的 ee_mask 属性，拼接成一个张量 target_mask，并将其移动到与 target_rot 相同的设备上；
#     # 从 data_list 中提取 ee_mask 拼接成 source_mask，也移至相同设备。
#     # 这些 mask 用于后续选择有效旋转信息进行损失计算。
#     target_mask = torch.cat([data.ee_mask for data in target_list]).to(target_rot.device)
#     # print("target_mask:",target_mask.size())
#     source_mask = torch.cat([data.ee_mask for data in data_list]).to(target_rot.device)
#     # print("source_mask:",source_mask.size())
#     # 该代码将输入的 target_rot 张量重塑为二维张量，
#     # 其中第二维的长度为 9，常用于表示展平后的旋转矩阵。
#     target_rot = target_rot.view(-1, 9)
#     # print("target_rot:",target_rot.size())
#     # 从 data_list 中提取所有 data.q（四元数）并拼接成一个张量；
#     # 将拼接后的四元数转换为旋转矩阵，使用函数 quaternion_to_rotation_matrix；
#     # 将得到的旋转矩阵展平为形状 (-1, 9)，即每个旋转矩阵变为长度为9的一维向量。
#     source_rot = quaternion_to_rotation_matrix(torch.cat([data.q for data in data_list]).to(target_rot.device)).view(-1,
#                                                                                                                      9)
#     # print("source_rot:",source_rot.size())
#
#     target_q = torch.masked_select(target_rot, target_mask)
#     # print("target_q:",target_q.size())
#     source_q = torch.masked_select(source_rot, source_mask)
#     # print("source_q:",source_q.size())
#     ori_loss = ori_criterion(target_q, source_q)
#     # 取绝对值
#     ori_loss = torch.abs(ori_loss)
#     return ori_loss

"""
Calculate Orientation Loss
"""
def calculate_ori_loss_dif(t_q_matrix, s_q_matrix, ori_criterion):
    """
    计算相邻帧之间的旋转矩阵差分，并使用给定的损失函数计算损失。
    
    参数:
    - data_list: list of torch_geometric.data.Data
    - target_list: list of torch_geometric.data.Data
    - t_q_matrix: torch.Tensor, shape [2*batch_size, 3, 3]
                  目标旋转矩阵
    - s_q_matrix: torch.Tensor, shape [2*batch_size, 3, 3]
                  源旋转矩阵
    - ori_criterion: 损失函数，如 nn.MSELoss()
    
    返回:
    - loss: 差分后的损失值
    """
    # Step 1: reshape 为 [-1 3, 3]
    t_rot = t_q_matrix.view(-1, 3, 3)  
    s_rot = s_q_matrix.view(-1, 3, 3)
    batch_size = t_rot.shape[0] // 2  # 假设 batch_size 是 t_rot 的第一维的一半
    # Step 2: 计算相邻帧之间的差分矩阵
    # 0 1是同一帧的左右手 2 3是下一帧的左右手
    #所以0与2做差分 1与3做差分 以此类推
    # 原始为 2*batch_size, 3, 3的数据
    # 得到 2*(batch_size - 1),3,3的数据
    # 差分用除法 : 除数为t_rot[0:2*batch_size-1, :, :] 被除数 t_rot[2:-1, :, :]
    left_rot_t = t_rot[::2]  # [0, 2]
    right_rot_t = t_rot[1::2]  # [1, 3]

    # 计算帧间差分
    left_diff_t = torch.bmm(left_rot_t[1:],left_rot_t[:-1].transpose(1, 2))  # 0→2
    right_diff_t = torch.bmm(right_rot_t[1:],right_rot_t[:-1].transpose(1, 2))  # 1→3
    diff_t = torch.cat((left_diff_t, right_diff_t), dim=0)  # 合并差分结果

    left_rot_s = s_rot[::2]  # [0, 2]
    right_rot_s = s_rot[1::2]  # [1, 3]
    # 计算帧间差分
    left_diff_s = torch.bmm(left_rot_s[1:], left_rot_s[:-1].transpose(1, 2))  # 0→2
    right_diff_s = torch.bmm(right_rot_s[1:], right_rot_s[:-1].transpose(1, 2))  # 1→3
    diff_s = torch.cat((left_diff_s, right_diff_s), dim=0)
    
    # 结果应该为[2*B-2, 3, 3]
    # Step 3: 展平差分矩阵以适配损失函数
    diff_t_flat = diff_t.view(-1, 9)  
    diff_s_flat = diff_s.view(-1, 9)

    # Step 5: 使用传入的 criterion 计算损失
    loss = ori_criterion(diff_t_flat, diff_s_flat)

    return loss
def calculate_ori_loss(data_list, target_list,target_rot,ori_criterion,target_pos,target_angle):
    target_mask = torch.cat([data.ee_mask for data in target_list]).to(target_rot.device)
    # print("target_mask:",target_mask.size())
    source_mask = torch.cat([data.ee_mask for data in data_list]).to(target_rot.device)
    # print("source_mask:",source_mask.size())
    target_rot = target_rot.view(-1, 9)
    # print("target_rot:",target_rot.size())
    source_rot = quaternion_to_rotation_matrix(torch.cat([data.q for data in data_list]).to(target_rot.device)).view(-1, 9)
    # print("source_rot:",source_rot.size())

    target_q = torch.masked_select(target_rot, target_mask)
    # print("target_q:",target_q.size())
    source_q = torch.masked_select(source_rot, source_mask)
    # print("source_q:",source_q.size())
    # print(target_q)
    # print(source_q)
    ori_loss = ori_criterion(target_q, source_q)

    # t_q_matrix = target_q.view(-1, 3, 3)
    # s_q_matrix = source_q.view(-1, 3, 3)
    # 计算相邻帧之间的旋转矩阵差分
    # ori_loss = calculate_ori_loss_dif(t_q_matrix,s_q_matrix, ori_criterion)
    # temp1= target_q
    # temp2 = source_q 
    # temp = temp1 - temp2
    return ori_loss

def calculate_ori_loss6(data_list, target_list,target_rot,ori_criterion,target_pos,target_angle):
    target_mask = torch.cat([data.ee_mask for data in target_list]).to(target_rot.device)
    # print("target_mask:",target_mask.size())
    source_mask = torch.cat([data.ee_mask for data in data_list]).to(target_rot.device)
    # print("source_mask:",source_mask.size())
    target_rot = target_rot.view(-1, 9)
    # print("target_rot:",target_rot.size())
    source_rot = quaternion_to_rotation_matrix(torch.cat([data.q for data in data_list]).to(target_rot.device)).view(-1, 9)
    target_q = torch.masked_select(target_rot, target_mask)
    # print("target_q:",target_q.size())
    source_q = torch.masked_select(source_rot, source_mask)
    # print("source_rot:",source_rot.size())
    #提取掌面向量
    source_vec = torch.cat([data.vec for data in data_list]).to(target_rot.device)
    #计算目标掌面向量：用旋转矩阵在左 初始向量(0,0,-1)在右计算得到实际掌面向量
    target_matrix = target_q.view(-1, 3, 3)
    #生成数量和 target_matrix一样 每个数据为(0,0,-1)的张量
    target_raw_vector = torch.tensor([1, 0, 0], dtype=target_matrix.dtype, device=target_matrix.device).repeat(target_matrix.size(0), 1)
    # target_raw_vector = torch.tensor([0, 0, -1], dtype=target_matrix.dtype, device=target_matrix.device).repeat(target_matrix.size(0), 1)
    
    #将偶数为左手，奇数为右手，将偶数位置为手心 右手手心
    # target_raw_vector[0::2] = -target_raw_vector[0::2]
    #逐个相乘
    target_vec = torch.bmm(target_matrix, target_raw_vector.view(-1, 3, 1)).view(-1, 3)
    # print("source_q:",source_q.size())
    # print(target_q)
    # print(source_q)
    # ori_loss = ori_criterion(target_q, source_q)
    ori_loss = ori_criterion(target_vec, source_vec)
    # t_q_matrix = target_q.view(-1, 3, 3)
    # s_q_matrix = source_q.view(-1, 3, 3)
    # 计算相邻帧之间的旋转矩阵差分
    # ori_loss = calculate_ori_loss_dif(t_q_matrix,s_q_matrix, ori_criterion)
    # temp1= target_q
    # temp2 = source_q 
    # temp = temp1 - temp2
    return ori_loss

"""
Calculate Orientation Loss
"""
def calculate_ori_loss1(data_list, target_list, l_hand_pos, r_hand_pos, ori_criterion):

    target_nv_mask = torch.cat([data.hand_nv_mask for data in target_list]).to(l_hand_pos.device)
    source_nv_mask = torch.cat([data.l_hand_nv_mask for data in data_list]).to(l_hand_pos.device)
    # print("source_nv_mask",source_nv_mask)
    # print("掩码形状:", source_nv_mask.shape)
    # print("True值数量:", source_nv_mask.sum().item())
    target_nv = torch.masked_select(l_hand_pos, target_nv_mask).view(-1, 3,3)
    source_nv = torch.masked_select(torch.cat([data.l_hand_pos for data in data_list]).to(l_hand_pos.device),
                                    source_nv_mask).view(-1, 3,3)
    # print("target_nv",target_nv)
    # print("source_nv",source_nv)
    target_nv_v1 = target_nv[:, 1, :] - target_nv[:, 0, :]  # index - wrist
    # print("target_nv_v1",target_nv_v1)
    target_nv_v2 = target_nv[:, 2, :] - target_nv[:, 0, :]  # ring - wrist
    # print("target_nv_v2",target_nv_v2)
    target_nv_normal = torch.cross(target_nv_v1, target_nv_v2, dim=-1)
    # print("left_target_target_nv_normal",target_nv_normal)
    source_nv_v1 = source_nv[:, 1, :] - source_nv[:, 0, :]
    source_nv_v2 = source_nv[:, 2, :] - source_nv[:, 0, :]
    source_nv_normal = torch.cross(source_nv_v1, source_nv_v2, dim=-1)
    # print("left_target_source_nv_normal",source_nv_normal)
    left_ori_loss = ori_criterion(target_nv_normal, source_nv_normal)
    # # 检查左手位置数据是否变化
    # for i, data in enumerate(data_list):
    #     print(f"Sample {i} l_hand_pos:", data.l_hand_pos)

    # # 检查掩码是否正确
    # # 在损失函数中添加验证
    # print("掩码形状:", target_nv_mask.shape)
    # print("True值数量:", target_nv_mask.sum().item())
    # print("选择后的点数:", target_nv_mask.sum().item() // 3)  # 应为batch_size

    target_nv_mask = torch.cat([data.hand_nv_mask for data in target_list]).to(r_hand_pos.device)
    source_nv_mask = torch.cat([data.r_hand_nv_mask for data in data_list]).to(r_hand_pos.device)
    target_nv = torch.masked_select(r_hand_pos, target_nv_mask).view(-1, 3, 3)
    source_nv = torch.masked_select(torch.cat([data.r_hand_pos for data in data_list]).to(r_hand_pos.device),
                                    source_nv_mask).view(-1, 3, 3)
    target_nv_v1 = target_nv[:, 1, :] - target_nv[:, 0, :]  # index - wrist
    target_nv_v2 = target_nv[:, 2, :] - target_nv[:, 0, :]  # ring - wrist
    target_nv_normal = torch.cross(target_nv_v1, target_nv_v2, dim=-1)

    source_nv_v1 = source_nv[:, 1, :] - source_nv[:, 0, :]
    source_nv_v2 = source_nv[:, 2, :] - source_nv[:, 0, :]
    source_nv_normal = torch.cross(source_nv_v1, source_nv_v2, dim=-1)
    right_ori_loss = ori_criterion(target_nv_normal, source_nv_normal)

    ori_loss = (left_ori_loss + right_ori_loss) / 2
    return ori_loss

def calculate_ori_loss2(data_list, target_list,target_rot,ori_criterion,target_pos,target_angle):
    """
    计算旋转损失
    通过末端两个关节的向量和末端关节的旋转角度计算末端效应器的旋转矩阵
    source的旋转矩阵是通过四元数转换得到的,与calculate_ori_loss函数中计算类似
    target的旋转矩阵是通过末端两个关节的向量和末端关节的旋转角度计算得到的,故
    需要两个关节的位置和一个关节的旋转角度
    """
    target_mask = torch.cat([data.ee_mask for data in target_list]).to(target_rot.device)
    # print("target_mask:",target_mask.size())
    source_mask = torch.cat([data.ee_mask for data in data_list]).to(target_rot.device)
    # print("source_mask:",source_mask.size())
    target_rot = target_rot.view(-1, 9)
    # print("target_rot:",target_rot.size())
    source_rot = quaternion_to_rotation_matrix(torch.cat([data.q for data in data_list]).to(target_rot.device)).view(-1, 9)
    # print("source_rot:",source_rot.size())
    target_q_raw = torch.masked_select(target_rot, target_mask)
    # print("target_q:",target_q.size())
    source_q = torch.masked_select(source_rot, source_mask)

    #计算另一个旋转矩阵
    target_ee_mask = torch.cat([data.ee_mask for data in target_list]).to(target_rot.device)
    target_ee = torch.masked_select(target_pos, target_ee_mask).view(-1, 3)
    #获取末端效应器前一个关节的位置
    target_el_mask = torch.cat([data.el_mask for data in target_list]).to(target_rot.device)
    target_el = torch.masked_select(target_pos, target_el_mask).view(-1, 3)
    #获取旋转轴
    target_axis = target_el-target_ee 
    # 获取末端效应器的角度
    target_angle_ee = torch.masked_select(target_angle, target_ee_mask)
    # 对张量的每一组进行计算末端效应器的旋转矩阵
    target_rot_matrix = torch.empty(target_ee.shape[0], 3, 3, device=target_rot.device)
    shape1 = target_angle_ee.shape
    shape2 = target_axis.shape
    shape3 = target_rot_matrix.shape
    shape4 = target_ee.shape 
    for i in range(target_ee.shape[0]):
        axis = target_axis[i]
        angle = target_angle_ee[i]
        target_rot_matrix[i] = get_rotation_matrix(axis, angle)
    
    shape5= target_rot_matrix.shape
    shape6 = source_q.shape
    shape7 = target_q_raw.shape
    target_rot_matrix_flat = target_rot_matrix.view(-1, 9).view(-1)
    shape8 = target_rot_matrix_flat.shape
    ori_loss = ori_criterion(target_q_raw, source_q)
    ori_loss_1 = ori_criterion(target_rot_matrix_flat, source_q)
    return ori_loss_1

def calculate_ori_loss3(data_list, target_list,target_rot,ori_criterion,target_pos,target_angle):
    """
    计算旋转损失
    通过直接比较末端效应器角度比较旋转损失
    """
    target_mask = torch.cat([data.ee_mask for data in target_list]).to(target_rot.device)
    # print("target_mask:",target_mask.size())
    source_mask = torch.cat([data.ee_mask for data in data_list]).to(target_rot.device)
    # print("source_mask:",source_mask.size())
    target_rot = target_rot.view(-1, 9)
    # print("target_rot:",target_rot.size())
    source_rot = quaternion_to_rotation_matrix(torch.cat([data.q for data in data_list]).to(target_rot.device)).view(-1, 9)
    # print("source_rot:",source_rot.size())
    target_q_raw = torch.masked_select(target_rot, target_mask)
    # print("target_q:",target_q.size())
    source_q = torch.masked_select(source_rot, source_mask)
    
    source_rot_raw = quaternion_to_rotation_matrix(torch.cat([data.q for data in data_list]).to(target_rot.device))
    source_ee_mask = torch.cat([data.ee_mask for data in data_list]).to(target_rot.device)
    source_el_mask = torch.cat([data.el_mask for data in data_list]).to(target_rot.device)
    source_ee = torch.masked_select(torch.cat([data.pos for data in data_list]).to(target_rot.device),
                                    source_ee_mask)
    source_el = torch.masked_select(torch.cat([data.pos for data in data_list]).to(target_pos.device),
                                    source_el_mask)
    source_axis = source_ee - source_el
    temp1=source_axis.shape
    temp2=source_rot_raw.shape
    source_axis = source_axis.view(-1, 3)
    source_angle = torch.empty(source_axis.shape[0], device=source_rot_raw.device)
    for i in range(source_axis.shape[0]):
        source_angle[i] = get_angle_from_rotation_matrix(source_rot_raw[i],source_axis[i])
    temp3=source_angle.shape
    target_ee_mask = torch.cat([data.ee_mask for data in target_list]).to(target_rot.device)
    # 获取末端效应器的角度
    target_angle_ee = torch.masked_select(target_angle, target_ee_mask)
    #限制角度在0-2*pi之间
    target_angle_ee = target_angle_ee % (2 * torch.pi)
    temp4=target_angle_ee.shape
    
    ori_loss = ori_criterion(target_q_raw, source_q)
    #取绝对值
    loss = torch.min(torch.abs(target_angle_ee - source_angle),2*torch.pi-torch.abs(target_angle_ee - source_angle))
    #新建一个张量和loss的形状相同全为0
    loss_baseline = torch.zeros_like(loss)
    ori_loss_1 = ori_criterion(loss, loss_baseline)
    return ori_loss_1

def calculate_ori_loss4(data_list, target_list,target_rot,ori_criterion,target_pos,target_angle):
    """
    计算旋转损失
    通过末端两个关节的向量和末端关节的旋转角度计算末端效应器的旋转矩阵
    source的旋转矩阵是通过四元数转换得到的,与calculate_ori_loss函数中计算类似
    target的旋转矩阵是通过末端两个关节的向量和末端关节的旋转角度计算得到的,故
    需要两个关节的位置和一个关节的旋转角度
    """
    target_mask = torch.cat([data.ee_mask for data in target_list]).to(target_rot.device)
    # print("target_mask:",target_mask.size())
    source_mask = torch.cat([data.ee_mask for data in data_list]).to(target_rot.device)
    # print("source_mask:",source_mask.size())
    target_rot = target_rot.view(-1, 9)
    # print("target_rot:",target_rot.size())
    ##elmask和shmask
    el_mask = torch.cat([data.el_mask for data in data_list]).to(target_rot.device)
    sh_mask = torch.cat([data.sh_mask for data in data_list]).to(target_rot.device)
    
    # print("el_rot:",el_rot.size())
    source_rot = quaternion_to_rotation_matrix(torch.cat([data.q for data in data_list]).to(target_rot.device)).view(-1, 9)
    
    el_rot = torch.masked_select(source_rot, el_mask)
    sh_rot = torch.masked_select(source_rot, sh_mask)
    ee_rot = torch.masked_select(source_rot, source_mask)
    source_q_raw = ee_rot
    #三组旋转矩阵相乘等于ee的全局旋转矩阵
    source_rot = torch.bmm(torch.bmm(sh_rot.view(-1, 3, 3),el_rot.view(-1, 3, 3)),ee_rot.view(-1, 3, 3))
    source_q = source_rot.view(-1, 9).view(-1)
    # print("source_rot:",source_rot.size())
    # print("source_mask:",source_mask.size())
    # source_rot = quaternion_to_rotation_matrix(torch.cat([data.q for data in data_list]).to(target_rot.device)).view(-1, 9)
    #不需要再次选择source_rot，因为已经在上面计算了
    

    target_q_raw = torch.masked_select(target_rot, target_mask)
    # print("target_q:",target_q.size())
    #计算另一个旋转矩阵
    target_ee_mask = torch.cat([data.ee_mask for data in target_list]).to(target_rot.device)
    target_ee = torch.masked_select(target_pos, target_ee_mask).view(-1, 3)
    #获取末端效应器前一个关节的位置
    target_el_mask = torch.cat([data.el_mask for data in target_list]).to(target_rot.device)
    target_el = torch.masked_select(target_pos, target_el_mask).view(-1, 3)
    #获取旋转轴
    target_axis = target_el-target_ee 
    # 获取末端效应器的角度
    target_angle_ee = torch.masked_select(target_angle, target_ee_mask)
    # 对张量的每一组进行计算末端效应器的旋转矩阵
    target_rot_matrix = torch.empty(target_ee.shape[0], 3, 3, device=target_rot.device)
    shape1 = target_angle_ee.shape
    shape2 = target_axis.shape
    shape3 = target_rot_matrix.shape
    shape4 = target_ee.shape 
    for i in range(target_ee.shape[0]):
        axis = target_axis[i]
        angle = target_angle_ee[i]
        target_rot_matrix[i] = get_rotation_matrix(axis, angle)
    shape5= target_rot_matrix.shape
    shape6 = source_q.shape
    shape7 = target_q_raw.shape
    target_rot_matrix_flat = target_rot_matrix.view(-1, 9).view(-1)
    shape8 = target_rot_matrix_flat.shape
    ori_loss = ori_criterion(target_q_raw, source_q_raw)
    ori_loss_1 = ori_criterion(target_rot_matrix_flat, source_q)
    return ori_loss_1

def calculate_ori_loss5(data_list, target_list,target_rot,ori_criterion,target_pos,target_angle):
    """
    计算旋转损失
    通过全局旋转矩阵计算损失
    """
    target_mask = torch.cat([data.ee_mask for data in target_list]).to(target_rot.device)
    # print("target_mask:",target_mask.size())
    source_mask = torch.cat([data.ee_mask for data in data_list]).to(target_rot.device)
    # print("source_mask:",source_mask.size())
    target_rot = target_rot.view(-1, 9)
    # print("target_rot:",target_rot.size())
    ##elmask和shmask
    el_mask = torch.cat([data.el_mask for data in data_list]).to(target_rot.device)
    sh_mask = torch.cat([data.sh_mask for data in data_list]).to(target_rot.device)
    
    # print("el_rot:",el_rot.size())
    source_rot = quaternion_to_rotation_matrix(torch.cat([data.q for data in data_list]).to(target_rot.device)).view(-1, 9)
    
    el_rot = torch.masked_select(source_rot, el_mask)
    sh_rot = torch.masked_select(source_rot, sh_mask)
    ee_rot = torch.masked_select(source_rot, source_mask)
    source_q_raw = ee_rot
    #三组旋转矩阵相乘等于ee的全局旋转矩阵
    source_rot = torch.bmm(torch.bmm(sh_rot.view(-1, 3, 3),el_rot.view(-1, 3, 3)),ee_rot.view(-1, 3, 3))
    source_q = source_rot.view(-1, 9).view(-1)

    # target_rot_trans = T @ target_rot.view(-1, 3, 3) @ T.T
    target_rot_trans = target_rot
    target_rot = target_rot_trans.view(-1, 9)
    target_q = torch.masked_select(target_rot, target_mask)
    # print("target_q:",target_q.size())
    
    # print("source_q:",source_q.size())
    # print(target_q)
    # print(source_q)
    ori_loss = ori_criterion(target_q, source_q)

    t_q_matrix = target_q.view(-1, 3, 3)
    s_q_matrix = source_q.view(-1, 3, 3)
    # temp1= target_q
    # temp2 = source_q 
    # temp = temp1 - temp2
    return ori_loss


"""
Calculate Finger Similarity Loss
"""
def calculate_fin_loss(data_list, target_list, l_hand_pos, r_hand_pos, ee_criterion):
    # left hand
    target_el_mask = torch.cat([data.hand_el_mask for data in target_list]).to(l_hand_pos.device)
    target_ee_mask = torch.cat([data.hand_ee_mask for data in target_list]).to(l_hand_pos.device)
    source_el_mask = torch.cat([data.l_hand_el_mask for data in data_list]).to(l_hand_pos.device)
    source_ee_mask = torch.cat([data.l_hand_ee_mask for data in data_list]).to(l_hand_pos.device)
    # print("source_el_mask",source_el_mask)
    # print("掩码形状:", source_el_mask.shape)
    # print("True值数量:", source_el_mask.sum().item())
    # print("source_ee_mask",source_ee_mask)
    target_el = torch.masked_select(l_hand_pos, target_el_mask).view(-1, 3)
    target_ee = torch.masked_select(l_hand_pos, target_ee_mask).view(-1, 3)
    source_el = torch.masked_select(torch.cat([data.l_hand_pos for data in data_list]).to(l_hand_pos.device),
                                    source_el_mask).view(-1, 3)
    # print("source_el",source_el)
    source_ee = torch.masked_select(torch.cat([data.l_hand_pos for data in data_list]).to(l_hand_pos.device),
                                    source_ee_mask).view(-1, 3)
    # print("target_el",target_el)
    # print("target_ee",target_ee)
    # print("source_ee",target_ee)
    target_vector = target_ee - target_el
    source_vector = source_ee - source_el
    # normalize
    target_elbow_dist = torch.cat([data.hand_elbow_dist for data in target_list]).to(l_hand_pos.device)
    source_elbow_dist = torch.cat([data.l_hand_elbow_dist for data in data_list]).to(l_hand_pos.device)
    normalize_target_vector = target_vector / torch.masked_select(target_elbow_dist, target_ee_mask).unsqueeze(1)
    normalize_source_vector = source_vector / torch.masked_select(source_elbow_dist, source_ee_mask).unsqueeze(1)
    # print("normalize_source_vector",normalize_source_vector)
    l_fin_loss = ee_criterion(normalize_target_vector, normalize_source_vector)

    # right hand
    target_el_mask = torch.cat([data.hand_el_mask for data in target_list]).to(r_hand_pos.device)
    target_ee_mask = torch.cat([data.hand_ee_mask for data in target_list]).to(r_hand_pos.device)
    source_el_mask = torch.cat([data.r_hand_el_mask for data in data_list]).to(r_hand_pos.device)
    source_ee_mask = torch.cat([data.r_hand_ee_mask for data in data_list]).to(r_hand_pos.device)
    target_el = torch.masked_select(r_hand_pos, target_el_mask).view(-1, 3)
    target_ee = torch.masked_select(r_hand_pos, target_ee_mask).view(-1, 3)
    source_el = torch.masked_select(torch.cat([data.r_hand_pos for data in data_list]).to(r_hand_pos.device),
                                    source_el_mask).view(-1, 3)
    source_ee = torch.masked_select(torch.cat([data.r_hand_pos for data in data_list]).to(r_hand_pos.device),
                                    source_ee_mask).view(-1, 3)
    target_vector = target_ee - target_el
    source_vector = source_ee - source_el
    # normalize
    target_elbow_dist = torch.cat([data.hand_elbow_dist for data in target_list]).to(r_hand_pos.device)
    # print("target_elbow_dist",target_elbow_dist)
    source_elbow_dist = torch.cat([data.r_hand_elbow_dist for data in data_list]).to(r_hand_pos.device)
    normalize_target_vector = target_vector / torch.masked_select(target_elbow_dist, target_ee_mask).unsqueeze(1)
    normalize_source_vector = source_vector / torch.masked_select(source_elbow_dist, source_ee_mask).unsqueeze(1)
    # torch.set_printoptions(threshold=torch.inf, precision=4, sci_mode=False)
    torch.set_printoptions(threshold=float('inf'), precision=4, sci_mode=False)
    # print('normalize_target_vector:',normalize_target_vector)
    # print('normalize_source_vector:',normalize_source_vector)
    r_fin_loss = ee_criterion(normalize_target_vector, normalize_source_vector)

    fin_loss = (l_fin_loss + r_fin_loss) / 2
    return fin_loss


"""
Collision Loss
"""


class CollisionLoss(nn.Module):
    def __init__(self, threshold, mode='capsule-capsule'):
        super(CollisionLoss, self).__init__()
        self.threshold = threshold
        self.mode = mode

    def forward(self, pos, edge_index, rot, ee_mask):
        """
        Keyword arguments:
        pos -- joint positions [batch_size, num_nodes, 3]
        edge_index -- edge index [2, num_edges]
        """
        batch_size = pos.shape[0]
        num_nodes = pos.shape[1]
        num_edges = edge_index.shape[1]

        # sphere-sphere detection
        if self.mode == 'sphere-sphere':
            l_sphere = pos[:, :num_nodes // 2, :]
            r_sphere = pos[:, num_nodes // 2:, :]
            l_sphere = l_sphere.unsqueeze(1).expand(batch_size, num_nodes // 2, num_nodes // 2, 3)
            r_sphere = r_sphere.unsqueeze(2).expand(batch_size, num_nodes // 2, num_nodes // 2, 3)
            dist_square = torch.sum(torch.pow(l_sphere - r_sphere, 2), dim=-1)
            mask = (dist_square < self.threshold ** 2) & (dist_square > 0)
            loss = torch.sum(torch.exp(-1 * torch.masked_select(dist_square, mask))) / batch_size

        # sphere-capsule detection
        if self.mode == 'sphere-capsule':
            # capsule p0 & p1
            p0 = pos[:, edge_index[0], :]
            p1 = pos[:, edge_index[1], :]
            # print(edge_index.shape, p0.shape, p1.shape)

            # left sphere & right capsule
            l_sphere = pos[:, :num_nodes // 2, :]
            r_capsule_p0 = p0[:, num_edges // 2:, :]
            r_capsule_p1 = p1[:, num_edges // 2:, :]
            dist_square_1 = self.sphere_capsule_dist_square(l_sphere, r_capsule_p0, r_capsule_p1, batch_size, num_nodes,
                                                            num_edges)

            # left capsule & right sphere
            r_sphere = pos[:, num_nodes // 2:, :]
            l_capsule_p0 = p0[:, :num_edges // 2, :]
            l_capsule_p1 = p1[:, :num_edges // 2, :]
            dist_square_2 = self.sphere_capsule_dist_square(r_sphere, l_capsule_p0, l_capsule_p1, batch_size, num_nodes,
                                                            num_edges)

            # calculate loss
            dist_square = torch.cat([dist_square_1, dist_square_2])
            mask = (dist_square < self.threshold ** 2) & (dist_square > 0)
            loss = torch.sum(torch.exp(-1 * torch.masked_select(dist_square, mask))) / batch_size

        # capsule-capsule detection
        if self.mode == 'capsule-capsule':
            # capsule p0 & p1
            p0 = pos[:, edge_index[0], :]
            p1 = pos[:, edge_index[1], :]
            # left capsule
            l_capsule_p0 = p0[:, :num_edges // 2, :]
            l_capsule_p1 = p1[:, :num_edges // 2, :]
            # right capsule
            r_capsule_p0 = p0[:, num_edges // 2:, :]
            r_capsule_p1 = p1[:, num_edges // 2:, :]
            # add capsule for left hand & right hand(for yumi)
            ee_pos = torch.masked_select(pos, ee_mask.to(pos.device)).view(-1, 3)
            ee_rot = torch.masked_select(rot, ee_mask.to(pos.device)).view(-1, 3, 3)
            offset = torch.Tensor([[[0], [0], [0.2]]]).repeat(ee_rot.size(0), 1, 1).to(pos.device)
            hand_pos = torch.bmm(ee_rot, offset).squeeze() + ee_pos
            l_ee_pos = ee_pos[::2, :].unsqueeze(1)
            l_hand_pos = hand_pos[::2, :].unsqueeze(1)
            r_ee_pos = ee_pos[1::2, :].unsqueeze(1)
            r_hand_pos = hand_pos[1::2, :].unsqueeze(1)
            l_capsule_p0 = torch.cat([l_capsule_p0, l_ee_pos], dim=1)
            l_capsule_p1 = torch.cat([l_capsule_p1, l_hand_pos], dim=1)
            r_capsule_p0 = torch.cat([r_capsule_p0, r_ee_pos], dim=1)
            r_capsule_p1 = torch.cat([r_capsule_p1, r_hand_pos], dim=1)
            num_edges += 2
            # print(l_capsule_p0.shape, l_capsule_p1.shape, r_capsule_p0.shape, r_capsule_p1.shape)
            # calculate loss
            dist_square = self.capsule_capsule_dist_square(l_capsule_p0, l_capsule_p1, r_capsule_p0, r_capsule_p1,
                                                           batch_size, num_edges)
            mask = (dist_square < 0.1 ** 2) & (dist_square > 0)
            mask[:, 4, 4] = (dist_square[:, 4, 4] < self.threshold ** 2) & (dist_square[:, 4, 4] > 0)
            loss = torch.sum(torch.exp(-1 * torch.masked_select(dist_square, mask))) / batch_size

        return loss

    def sphere_capsule_dist_square(self, sphere, capsule_p0, capsule_p1, batch_size, num_nodes, num_edges):
        # condition 1: p0 is the closest point
        vec_p0_p1 = capsule_p1 - capsule_p0  # vector p0-p1 [batch_size, num_edges//2, 3]
        vec_p0_pr = sphere.unsqueeze(2).expand(batch_size, num_nodes // 2, num_edges // 2, 3) - capsule_p0.unsqueeze(
            1).expand(batch_size, num_nodes // 2, num_edges // 2,
                      3)  # vector p0-pr [batch_size, num_nodes//2, num_edges//2, 3]
        vec_mul_p0 = torch.mul(vec_p0_p1.unsqueeze(1).expand(batch_size, num_nodes // 2, num_edges // 2, 3),
                               vec_p0_pr).sum(
            dim=-1)  # vector p0-p1 * vector p0-pr [batch_size, num_nodes//2, num_edges//2]
        dist_square_p0 = torch.masked_select(vec_p0_pr.norm(dim=-1) ** 2, vec_mul_p0 <= 0)
        # print(dist_square_p0.shape)

        # condition 2: p1 is the closest point
        vec_p1_p0 = capsule_p0 - capsule_p1  # vector p1-p0 [batch_size, num_edges//2, 3]
        vec_p1_pr = sphere.unsqueeze(2).expand(batch_size, num_nodes // 2, num_edges // 2, 3) - capsule_p1.unsqueeze(
            1).expand(batch_size, num_nodes // 2, num_edges // 2,
                      3)  # vector p1-pr [batch_size, num_nodes//2, num_edges//2, 3]
        vec_mul_p1 = torch.mul(vec_p1_p0.unsqueeze(1).expand(batch_size, num_nodes // 2, num_edges // 2, 3),
                               vec_p1_pr).sum(
            dim=-1)  # vector p1-p0 * vector p1-pr [batch_size, num_nodes//2, num_edges//2]
        dist_square_p1 = torch.masked_select(vec_p1_pr.norm(dim=-1) ** 2, vec_mul_p1 <= 0)
        # print(dist_square_p1.shape)

        # condition 3: closest point in p0-p1 segement
        d = vec_mul_p0 / vec_p0_p1.norm(dim=-1).unsqueeze(1).expand(batch_size, num_nodes // 2,
                                                                    num_edges // 2)  # vector p0-p1 * vector p0-pr / |vector p0-p1| [batch_size, num_nodes//2, num_edges//2]
        dist_square_middle = vec_p0_pr.norm(
            dim=-1) ** 2 - d ** 2  # distance square [batch_size, num_nodes//2, num_edges//2]
        dist_square_middle = torch.masked_select(dist_square_middle, (vec_mul_p0 > 0) & (vec_mul_p1 > 0))
        # print(dist_square_middle.shape)

        return torch.cat([dist_square_p0, dist_square_p1, dist_square_middle])

    def capsule_capsule_dist_square(self, capsule_p0, capsule_p1, capsule_q0, capsule_q1, batch_size, num_edges):
        # expand left capsule
        capsule_p0 = capsule_p0.unsqueeze(1).expand(batch_size, num_edges // 2, num_edges // 2, 3)
        capsule_p1 = capsule_p1.unsqueeze(1).expand(batch_size, num_edges // 2, num_edges // 2, 3)
        # expand right capsule
        capsule_q0 = capsule_q0.unsqueeze(2).expand(batch_size, num_edges // 2, num_edges // 2, 3)
        capsule_q1 = capsule_q1.unsqueeze(2).expand(batch_size, num_edges // 2, num_edges // 2, 3)
        # basic variables
        a = torch.mul(capsule_p1 - capsule_p0, capsule_p1 - capsule_p0).sum(dim=-1)
        b = torch.mul(capsule_p1 - capsule_p0, capsule_q1 - capsule_q0).sum(dim=-1)
        c = torch.mul(capsule_q1 - capsule_q0, capsule_q1 - capsule_q0).sum(dim=-1)
        d = torch.mul(capsule_p1 - capsule_p0, capsule_p0 - capsule_q0).sum(dim=-1)
        e = torch.mul(capsule_q1 - capsule_q0, capsule_p0 - capsule_q0).sum(dim=-1)
        f = torch.mul(capsule_p0 - capsule_q0, capsule_p0 - capsule_q0).sum(dim=-1)
        # initialize s, t to zero
        s = torch.zeros(batch_size, num_edges // 2, num_edges // 2).to(capsule_p0.device)
        t = torch.zeros(batch_size, num_edges // 2, num_edges // 2).to(capsule_p0.device)
        one = torch.ones(batch_size, num_edges // 2, num_edges // 2).to(capsule_p0.device)
        # calculate coefficient
        det = a * c - b ** 2
        bte = b * e
        ctd = c * d
        ate = a * e
        btd = b * d
        # nonparallel segments
        # region 6
        s = torch.where((det > 0) & (bte <= ctd) & (e <= 0) & (-d >= a), one, s)
        s = torch.where((det > 0) & (bte <= ctd) & (e <= 0) & (-d < a) & (-d > 0), -d / a, s)
        # region 5
        t = torch.where((det > 0) & (bte <= ctd) & (e > 0) & (e < c), e / c, t)
        # region 4
        s = torch.where((det > 0) & (bte <= ctd) & (e > 0) & (e >= c) & (b - d >= a), one, s)
        s = torch.where((det > 0) & (bte <= ctd) & (e > 0) & (e >= c) & (b - d < a) & (b - d > 0), (b - d) / a, s)
        t = torch.where((det > 0) & (bte <= ctd) & (e > 0) & (e >= c), one, t)
        # region 8
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e <= 0) & (-d > 0) & (-d < a), -d / a, s)
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e <= 0) & (-d > 0) & (-d >= a), one, s)
        # region 1
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e > 0) & (b + e < c), one, s)
        t = torch.where((det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e > 0) & (b + e < c), (b + e) / c, t)
        # region 2
        s = torch.where(
            (det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e > 0) & (b + e >= c) & (b - d > 0) & (b - d < a),
            (b - d) / a, s)
        s = torch.where(
            (det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e > 0) & (b + e >= c) & (b - d > 0) & (b - d >= a), one,
            s)
        t = torch.where((det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e > 0) & (b + e >= c), one, t)
        # region 7
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd < det) & (ate <= btd) & (-d > 0) & (-d >= a), one, s)
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd < det) & (ate <= btd) & (-d > 0) & (-d < a), -d / a, s)
        # region 3
        s = torch.where(
            (det > 0) & (bte > ctd) & (bte - ctd < det) & (ate > btd) & (ate - btd >= det) & (b - d > 0) & (b - d >= a),
            one, s)
        s = torch.where(
            (det > 0) & (bte > ctd) & (bte - ctd < det) & (ate > btd) & (ate - btd >= det) & (b - d > 0) & (b - d < a),
            (b - d) / a, s)
        t = torch.where((det > 0) & (bte > ctd) & (bte - ctd < det) & (ate > btd) & (ate - btd >= det), one, t)
        # region 0
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd < det) & (ate > btd) & (ate - btd < det),
                        (bte - ctd) / det, s)
        t = torch.where((det > 0) & (bte > ctd) & (bte - ctd < det) & (ate > btd) & (ate - btd < det),
                        (ate - btd) / det, t)
        # parallel segments
        # e <= 0
        s = torch.where((det <= 0) & (e <= 0) & (-d > 0) & (-d >= a), one, s)
        s = torch.where((det <= 0) & (e <= 0) & (-d > 0) & (-d < a), -d / a, s)
        # e >= c
        s = torch.where((det <= 0) & (e > 0) & (e >= c) & (b - d > 0) & (b - d >= a), one, s)
        s = torch.where((det <= 0) & (e > 0) & (e >= c) & (b - d > 0) & (b - d < a), (b - d) / a, s)
        t = torch.where((det <= 0) & (e > 0) & (e >= c), one, t)
        # 0 < e < c
        t = torch.where((det <= 0) & (e > 0) & (e < c), e / c, t)
        # print(s, t)
        s = s.unsqueeze(-1).expand(batch_size, num_edges // 2, num_edges // 2, 3).detach()
        t = t.unsqueeze(-1).expand(batch_size, num_edges // 2, num_edges // 2, 3).detach()
        w = capsule_p0 - capsule_q0 + s * (capsule_p1 - capsule_p0) - t * (capsule_q1 - capsule_q0)
        dist_square = torch.mul(w, w).sum(dim=-1)
        return dist_square


"""
Joint Limit Loss
"""


class JointLimitLoss(nn.Module):
    def __init__(self):
        super(JointLimitLoss, self).__init__()

    def forward(self, ang, lower, upper):
        """
        Keyword auguments:
        ang -- joint angles [batch_size*num_nodes, num_node_features]
        """
        # calculate mask with limit
        lower_mask = ang < lower
        upper_mask = ang > upper

        # calculate final loss
        lower_loss = torch.sum(torch.masked_select(lower - ang, lower_mask))
        upper_loss = torch.sum(torch.masked_select(ang - upper, upper_mask))
        loss = (lower_loss + upper_loss) / ang.shape[0]

        return loss


"""
Regularization Loss
"""


class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, z):
        # calculate final loss
        batch_size = z.shape[0]
        loss = torch.mean(torch.norm(z.view(batch_size, -1), dim=1).pow(2))

        return loss


def get_rotation_matrix(axis: torch.Tensor, angle: float, mode: str = 'rad') -> torch.Tensor:
    """
    根据旋转轴和旋转角度计算旋转矩阵 (PyTorch 实现)。
    该函数会自动将结果张量放置在与输入 `axis` 张量相同的设备上。

    参数:
    axis (torch.Tensor): 3D 旋转轴向量，形状为 (3,)。
    angle (float or torch.Tensor): 旋转角度。
    mode (str): 角度单位，'rad' (弧度) 或 'deg' (度数)。默认为 'rad'。

    返回:
    torch.Tensor: 3x3 的旋转矩阵。
    """
    # --- 1. 参数预处理和设备/类型获取 ---
    # 确保 axis 是一个浮点类型的 tensor
    if not isinstance(axis, torch.Tensor):
        axis = torch.tensor(axis, dtype=torch.float32)
    
    if axis.ndim != 1 or axis.shape[0] != 3:
        raise ValueError("axis 必须是一个包含3个元素的向量")

    # 获取输入张量的设备和数据类型，以确保所有计算都在同一设备上进行
    device = axis.device
    dtype = axis.dtype

    # 将角度转换为张量，并放置在同一设备上
    if not isinstance(angle, torch.Tensor):
        angle_tensor = torch.tensor(angle, device=device, dtype=dtype)
    else:
        angle_tensor = angle.to(device=device, dtype=dtype)
        
    if mode != 'rad':
        angle_rad = torch.deg2rad(angle_tensor)
    else:
        angle_rad = angle_tensor
    
    # --- 2. 向量和矩阵准备 ---
    # 将旋转轴向量单位化 (增加一个极小值 epsilon 防止除以零)
    axis_normalized = axis / (torch.linalg.norm(axis) + 1e-8)
    
    # 提取单位向量的分量
    kx, ky, kz = axis_normalized[0], axis_normalized[1], axis_normalized[2]
    
    # 计算 sin 和 cos
    c = torch.cos(angle_rad)
    s = torch.sin(angle_rad)
    
    # 创建单位矩阵 I 和叉乘矩阵 K
    I = torch.eye(3, device=device, dtype=dtype)
    K = torch.tensor([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ], device=device, dtype=dtype)
    
    # --- 3. 根据罗德里格斯公式构建矩阵 ---
    # R = I*cos(t) + (1-cos(t))*k*k^T + K*sin(t)
    # 另一种等价且常用的形式是: R = I + sin(t)*K + (1-cos(t))*K^2
    
    n = axis_normalized.unsqueeze(1)  # 转换为列向量
    nT =torch.transpose(n, 0, 1)
    K_sq = torch.matmul(n,nT)
    R = c*I + s * K + (1 - c) * K_sq
    
    return R

# 之前的函数 get_rotation_matrix 重命名为 get_rotation_matrix_from_axis_angle
# 以便与新函数区分。代码保持不变。
def get_rotation_matrix_from_axis_angle(axis: torch.Tensor, angle: float, mode: str = 'rad') -> torch.Tensor:
    """
    根据旋转轴和旋转角度计算旋转矩阵 (PyTorch 实现)。
    """
    # --- 1. 参数预处理和设备/类型获取 ---
    if not isinstance(axis, torch.Tensor):
        axis = torch.tensor(axis, dtype=torch.float32)
    if axis.ndim != 1 or axis.shape[0] != 3:
        raise ValueError("axis 必须是一个包含3个元素的向量")
    device, dtype = axis.device, axis.dtype
    if not isinstance(angle, torch.Tensor):
        angle_tensor = torch.tensor(angle, device=device, dtype=dtype)
    else:
        angle_tensor = angle.to(device=device, dtype=dtype)
    if mode != 'rad':
        angle_rad = torch.deg2rad(angle_tensor)
    else:
        angle_rad = angle_tensor
    
    # --- 2. 向量和矩阵准备 ---
    axis_normalized = axis / (torch.linalg.norm(axis) + 1e-8)
    kx, ky, kz = axis_normalized[0], axis_normalized[1], axis_normalized[2]
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    I = torch.eye(3, device=device, dtype=dtype)
    K = torch.tensor([[0, -kz, ky],
                     [kz, 0, -kx],
                     [-ky, kx, 0]], device=device, dtype=dtype)
    
    # --- 3. 根据罗德里格斯公式构建矩阵 ---
    #生成n*nT
    n = axis_normalized.unsqueeze(1)  # 转换为列向量
    K_sq = torch.matmul(n,torch.transpose(n, 0, 1))
    R = c*I + s * K + (1 - c) * K_sq
    return R

def get_angle_from_rotation_matrix(R: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    根据旋转矩阵和旋转轴反推出旋转角度 (弧度)。

    参数:
    R (torch.Tensor): 3x3 的旋转矩阵。
    axis (torch.Tensor): 3D 旋转轴向量，形状为 (3,)。

    返回:
    torch.Tensor: 旋转角度 (弧度)，一个标量张量。
    """
    # --- 1. 输入验证和设备/类型获取 ---
    if R.shape != (3, 3):
        raise ValueError("R 必须是一个 3x3 的旋转矩阵")
    if axis.ndim != 1 or axis.shape[0] != 3:
        raise ValueError("axis 必须是一个包含3个元素的向量")

    device, dtype = R.device, R.dtype
    axis = axis.to(device=device, dtype=dtype)
    
    # --- 2. 计算 cos(theta) ---
    # 单位化旋转轴
    axis_normalized = axis / (torch.linalg.norm(axis) + 1e-8)
    
    # trace = 1 + 2*cos(theta)
    trace = torch.trace(R)
    # 为防止浮点误差导致值超出[-1, 1]范围，进行裁剪
    cos_theta = torch.clamp((trace - 1) / 2, -1.0, 1.0)

    # --- 3. 计算 sin(theta) ---
    # sin(theta)k = (R - R^T) / 2
    # 通过与 k 点乘来提取 sin(theta)
    k = axis_normalized
    sin_theta_vec = torch.tensor([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], device=device, dtype=dtype)
    sin_theta = torch.dot(sin_theta_vec, k) / 2
    
    # --- 4. 使用 atan2 计算最终角度 ---
    angle_rad = torch.atan2(sin_theta, cos_theta)
    # 将角度限制在 [0, 2*pi] 范围内
    angle_rad = angle_rad % (2 * torch.pi)
    return angle_rad

if __name__ == '__main__':
    # 自动选择设备 (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用的设备: {device}\n")

    # --- 测试用例 ---
    # 1. 定义一个任意的旋转轴和角度
    axis_test = torch.tensor([1.0, 2.0, 3.0], device=device)
    angle_test_deg = 75.0
    angle_test_rad_true = torch.deg2rad(torch.tensor(angle_test_deg, device=device))

    print(f"原始轴: {axis_test.cpu().numpy()}")
    print(f"原始角度 (弧度): {angle_test_rad_true.item():.6f}")
    
    # 2. 使用正向函数生成旋转矩阵
    R_generated = get_rotation_matrix_from_axis_angle(axis_test, angle_test_deg, mode='deg')
    
    # 3. 使用反向函数从矩阵和轴恢复角度
    angle_recovered_rad = get_angle_from_rotation_matrix(R_generated, axis_test)
    
    print(f"从矩阵恢复的角度 (弧度): {angle_recovered_rad.item():.6f}\n")
    
    # 验证恢复的角度是否与原始角度接近
    assert torch.allclose(angle_test_rad_true, angle_recovered_rad), "恢复的角度不正确！"
    print("测试通过：恢复的角度与原始角度一致。")

    print("-" * 40)

    # --- 测试特殊情况：180度旋转 ---
    angle_180_deg = 180.0
    angle_180_rad_true = torch.pi
    R_180 = get_rotation_matrix_from_axis_angle(axis_test, angle_180_deg, mode='deg')
    angle_180_recovered = get_angle_from_rotation_matrix(R_180, axis_test)
    print(f"测试 180 度旋转 (π = {torch.pi:.6f})")
    print(f"恢复的角度: {angle_180_recovered.item():.6f}")
    # assert torch.allclose(torch.tensor(angle_180_rad_true), angle_180_recovered), "180度测试失败"
    print("180度旋转测试通过。")

# if __name__ == '__main__':
#     fake_sphere = torch.Tensor([[[2, 3, 0]]])
#     fake_capsule_p0 = torch.Tensor([[[0, 0, 0]]])
#     fake_capsule_p1 = torch.Tensor([[[1, 0, 0]]])
#     col_loss = CollisionLoss(threshold=1.0)
#     # print(col_loss.sphere_capsule_dist_square(fake_sphere, fake_capsule_p0, fake_capsule_p1, 1, 2, 2))
#     fake_capsule_p0 = torch.Tensor([[[0, 0, 0]]])
#     fake_capsule_p1 = torch.Tensor([[[1, 0, 0]]])
#     fake_capsule_q0 = torch.Tensor([[[-10, 0, 0]]])
#     fake_capsule_q1 = torch.Tensor([[[-9, 2, 0]]])
#     print(
#         col_loss.capsule_capsule_dist_square(fake_capsule_p0, fake_capsule_p1, fake_capsule_q0, fake_capsule_q1, 1, 2))
