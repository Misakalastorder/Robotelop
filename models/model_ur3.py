import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from kinematics import ForwardKinematicsURDF, ForwardKinematicsAxis


class SpatialBasicBlock(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, aggr='add', batch_norm=False, bias=True, **kwargs):
        super(SpatialBasicBlock, self).__init__(aggr=aggr, **kwargs)
        self.batch_norm = batch_norm
        # network architecture
        self.lin = nn.Linear(2*in_channels + edge_channels, out_channels, bias=bias)
        self.upsample = nn.Linear(in_channels, out_channels, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        # print("SpatialBasicBlock_x-- size:", x.size())
        if isinstance(x, torch.Tensor):
            x = (x, x)
        # print("SpatialBasicBlock_x[1] size:", x[1].size())
        # print("SpatialBasicBlock_self.upsample input size:", self.upsample.in_features)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # print("---------------out1 size:", out.size())
        out = self.bn(out) if self.batch_norm else out
        # print("---------------out2 size:", out.size())
        out += self.upsample(x[1])
        # print("---------------out2 upsample:", x[1].size())
        # print("---------------out3 size:", out.size())
        return out

    def message(self, x_i, x_j, edge_attr):
        # print("message_xi_size", x_i.size())
        # print("message_xj_size", x_j.size())
        # print("message_edge_attr_size", edge_attr.size())
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        # print("message_z", z.size())
        return F.leaky_relu(self.lin(z))


class Encoder(torch.nn.Module):
    def __init__(self, channels, dim):
        super(Encoder, self).__init__()
        self.conv1 = SpatialBasicBlock(in_channels=channels, out_channels=16, edge_channels=dim)
        self.conv2 = SpatialBasicBlock(in_channels=16, out_channels=32, edge_channels=dim)
        self.conv3 = SpatialBasicBlock(in_channels=32, out_channels=64, edge_channels=dim)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        """

        out = self.conv1(x, edge_index, edge_attr)
        out = self.conv2(out, edge_index, edge_attr)
        out = self.conv3(out, edge_index, edge_attr)
        return out

#+2的含义是加上lower upper
class Decoder(torch.nn.Module):
    def __init__(self, channels, dim):
        super(Decoder, self).__init__()
        self.conv1 = SpatialBasicBlock(in_channels=64+2, out_channels=32, edge_channels=dim)
        self.conv2 = SpatialBasicBlock(in_channels=32, out_channels=16, edge_channels=dim)
        self.conv3 = SpatialBasicBlock(in_channels=16, out_channels=channels, edge_channels=dim)

    def forward(self, x, edge_index, edge_attr, lower, upper):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        """
        # print("------------------------------")
        # print("-------------x",x.size())
        # print("-------------lower",lower.size())
        # print("-------------upper",upper.size())
        # print("------------------------------")
        # x_part = x[:lower.size(0), :]
        # x = torch.cat([x_part, lower, upper], dim=1)

        x = torch.cat([x, lower, upper], dim=1)
        # print("x", x.size())
        out = self.conv1(x, edge_index, edge_attr)
        # print("out", out.size())
        out = self.conv2(out, edge_index, edge_attr)
        # print("out", out.size())
        out = self.conv3(out, edge_index, edge_attr).tanh()
        # print("out", out.size())
        return out


class ArmNet(torch.nn.Module):
    def __init__(self):
        super(ArmNet, self).__init__()
        self.encoder = Encoder(6, 3)
        self.transform = nn.Sequential(
            nn.Linear(6*64, 12*64),
            nn.Tanh(),
        )
        self.decoder = Decoder(1, 6)
        self.fk = ForwardKinematicsAxis()
        # self.fk = ForwardKinematicsURDF()
    
    def forward(self, data, target):
        return self.decode(self.encode(data), target)
    
    def encode(self, data):
        # print("encode-------------data.x:", data.x.size())
        z = self.encoder(data.x, data.edge_index, data.edge_attr)
        # print("encode-------------z_size:", z.size())
        # print("encode-------------num_graphs:",data.num_graphs)
        z = self.transform(z.view(data.num_graphs, -1, 64).view(data.num_graphs, -1)).view(data.num_graphs, -1, 64).view(-1, 64)
        # print("encode-------------z_change_size:", z.size())
        return z
    
    def decode(self, z, target):
        # print("decode-------------z.x:", z.size())
        # print("decode-------------target.x:", (target.edge_index).size())

        ang = self.decoder(z, target.edge_index, target.edge_attr, target.lower, target.upper)
        # print("decode-------------ang1.x:", ang.size())

        ang = target.lower + (target.upper - target.lower)*(ang + 1)/2
        # print("decode-------------ang2.x:", ang.size())
        # print("decode------------target.parent",(target.parent).size())
        # print("decode------------target.offset",(target.offset).size())
        # print("decode------------target.num_graphs",(target.num_graphs))2

        pos, rot, global_pos = self.fk(ang, target.parent, target.offset, target.num_graphs,target.axis)
        # pos, rot, global_pos = self.fk(ang, target.parent, target.offset, target.num_graphs)
        # print(22222222222222222222)
        # print(pos.size())
        # print(33333333333333333333)
        # print(rot.size())
        # print(44444444444444444444)
        # print(global_pos.size())
        return z, ang, pos, rot, global_pos, None, None, None, None



class HandNet(torch.nn.Module):
    def __init__(self):
        super(HandNet, self).__init__()
        self.encoder = Encoder(3, 3)
        self.transform = nn.Sequential(
            nn.Linear(17*64, 18*64),
            nn.Tanh(),
        )
        self.decoder = Decoder(1, 6)
        self.fk = ForwardKinematicsAxis()

    def forward(self, data, target):
        return self.decode(self.encode(data), target)

    def encode(self, data):
        # print("l_hand_x size:", data.l_hand_x.size())
        # print("r_hand_x size:", data.r_hand_x.size())
        x = torch.cat([data.l_hand_x, data.r_hand_x], dim=0)
        edge_index = torch.cat([data.l_hand_edge_index, data.r_hand_edge_index+data.l_hand_x.size(0)], dim=1)
        # print("encode**************************data_size",(data.l_hand_edge_index).size(),(data.r_hand_edge_index).size(),(data.l_hand_x.size(0)))
        edge_attr = torch.cat([data.l_hand_edge_attr, data.r_hand_edge_attr], dim=0)

        z = self.encoder(x, edge_index, edge_attr)
        # print("encode-------------z_hand_size:", z.size())
        # print("encode-------------num_hand_graphs:",data.num_graphs)
        z = self.transform(z.view(2*data.num_graphs, -1, 64).view(2*data.num_graphs, -1)).view(2*data.num_graphs, -1, 64).view(-1, 64)
        # print("encode-------------z_hand_change_size:", z.size())
        # l_hand_z = self.encoder(data.l_hand_x, data.l_hand_edge_index, data.l_hand_edge_attr)
        # l_hand_z = self.transform(l_hand_z.view(data.num_graphs, -1, 64).view(data.num_graphs, -1)).view(data.num_graphs, -1, 64).view(-1, 64)
        # r_hand_z = self.encoder(data.r_hand_x, data.r_hand_edge_index, data.r_hand_edge_attr)
        # r_hand_z = self.transform(r_hand_z.view(data.num_graphs, -1, 64).view(data.num_graphs, -1)).view(data.num_graphs, -1, 64).view(-1, 64)
        # z = torch.cat([l_hand_z, r_hand_z], dim=0)
        return z

    def decode(self, z, target):
        # edge_index = torch.cat([target.hand_edge_index, target.hand_edge_index+z.size(0)//2], dim=1)
        edge_index = torch.cat([target.hand_edge_index, target.hand_edge_index], dim=1)
        # print("decode-------------edge_index:", edge_index.size())
        edge_attr = torch.cat([target.hand_edge_attr, target.hand_edge_attr], dim=0)
        # print("decode-------------edge_attr:", edge_attr.size())
        lower = torch.cat([target.hand_lower, target.hand_lower], dim=0)
        # print("decode-------------lower_size:", lower.size())
        upper = torch.cat([target.hand_upper, target.hand_upper], dim=0)
        # print("decode-------------upper_size:", upper.size())
        offset = torch.cat([target.hand_offset, target.hand_offset], dim=0)
        # print("decode-------------offset_size:", offset.size())
        parent = torch.cat([target.hand_parent, target.hand_parent], dim=0)
        # print("decode-------------parent:", parent.size())
        num_graphs = 2*target.num_graphs
        # print("decode-------------num_graphs:", num_graphs)
        axis = torch.cat([target.hand_axis, target.hand_axis], dim=0)
        # print("decode-------------axis:", axis.size())
        hand_ang = self.decoder(z, edge_index, edge_attr, lower, upper)
        # print("decode-------------hand_ang:", hand_ang.size())

        hand_ang = lower + (upper - lower)*(hand_ang + 1)/2
        hand_pos, _, _ = self.fk(hand_ang, parent, offset, num_graphs, axis)
        # print("decode-------------hand_pos:", hand_pos.size())
        half = hand_ang.size(0)//2
        l_hand_ang, r_hand_ang = hand_ang[:half, :], hand_ang[half:, :]
        l_hand_pos, r_hand_pos = hand_pos[:half, :], hand_pos[half:, :]


        # half = z.shape[0] // 2
        # l_hand_z, r_hand_z = z[:half, :], z[half:, :]

        # l_hand_ang = self.decoder(l_hand_z, target.hand_edge_index, target.hand_edge_attr, target.hand_lower, target.hand_upper)
        # l_hand_ang = target.hand_lower + (target.hand_upper - target.hand_lower)*(l_hand_ang + 1)/2
        # l_hand_pos, _, _ = self.fk(l_hand_ang, target.hand_parent, target.hand_offset, target.num_graphs, target.hand_axis)

        # r_hand_ang = self.decoder(r_hand_z, target.hand_edge_index, target.hand_edge_attr, target.hand_lower, target.hand_upper)
        # r_hand_ang = target.hand_lower + (target.hand_upper - target.hand_lower)*(r_hand_ang + 1)/2
        # r_hand_pos, _, _ = self.fk(r_hand_ang, target.hand_parent, target.hand_offset, target.num_graphs, target.hand_axis)
        return z, None, None, None, None, l_hand_ang, l_hand_pos, r_hand_ang, r_hand_pos


class YumiNet(torch.nn.Module):
    def __init__(self):
        super(YumiNet, self).__init__()
        self.arm_net = ArmNet()
        self.hand_net = HandNet()

    def forward(self, data, target):
        return self.decode(self.encode(data), target)

    def encode(self, data):
        # print("11111111111111111")
        arm_z = self.arm_net.encode(data)
        # print("22222222222222222")
        hand_z = self.hand_net.encode(data)
        z = torch.cat([arm_z, hand_z], dim=0)
        return z

    def decode(self, z, target):
        half = target.num_nodes
        # print("decode------------half",half)
        arm_z, hand_z = z[:half, :], z[half:, :]
        # print("333333333333333333")
        _, ang, pos, rot, global_pos, _, _, _, _ = self.arm_net.decode(arm_z, target)
        # print("444444444444444444")
        _, _, _, _, _, l_hand_ang, l_hand_pos, r_hand_ang, r_hand_pos = self.hand_net.decode(hand_z, target)
        return z, ang, pos, rot, global_pos, l_hand_ang, l_hand_pos, r_hand_ang, r_hand_pos

# DataBatch(x=[144, 1], edge_index=[2, 120], edge_attr=[120, 6], skeleton_type=[12], topology_type=[12], 
#           ee_mask=[144, 1], sh_mask=[144, 1], el_mask=[144, 1], root_dist=[144, 1], shoulder_dist=[144, 1], 
#           elbow_dist=[144, 1], num_nodes=144, parent=[144], offset=[144, 6], axis=[144, 3], lower=[144, 1],
#             upper=[144, 1], 
# 
# hand_x=[156, 1], hand_edge_index=[2, 144], hand_edge_attr=[144, 6], 
#             hand_ee_mask=[156, 1], hand_el_mask=[156, 1], hand_nv_mask=[156, 1], hand_root_dist=[156, 1],
#               hand_elbow_dist=[156, 1], hand_num_nodes=[12], hand_parent=[156], hand_offset=[156, 6], 
#           hand_axis=[156, 3], 
# 
# hand_lower=[156, 1], hand_upper=[156, 1], batch=[144], ptr=[13])

# DataBatch(x=[144, 1], edge_index=[2, 120], edge_attr=[120, 6], 
#           skeleton_type=[12], topology_type=[12], 
#           ee_mask=[144, 1], sh_mask=[144, 1], el_mask=[144, 1], 
#           root_dist=[144, 1], shoulder_dist=[144, 1], 
#           elbow_dist=[144, 1], num_nodes=144, parent=[144], 
#           offset=[144, 6], axis=[144, 3], lower=[144, 1], upper=[144, 1], 
#           hand_x=[156, 1], hand_edge_index=[2, 144], hand_edge_attr=[144, 6],
#           hand_ee_mask=[156, 1], hand_el_mask=[156, 1], hand_nv_mask=[156, 1],
#             hand_root_dist=[156, 1], hand_elbow_dist=[156, 1], 
#             hand_num_nodes=[12], hand_parent=[156], hand_offset=[156, 6], 
#             hand_axis=[156, 3],
#            hand_lower=[156, 1], hand_upper=[156, 1], batch=[144], ptr=[13])