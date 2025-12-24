import torch#导入torch
import torch.nn as nn#导入神经网络
import torch.optim as optim#导入优化器
import torch_geometric.transforms as transforms
from torch_geometric.loader import DataListLoader
from tensorboardX import SummaryWriter#导入训练显示器
# from torch_geometric.data import Batch

from models import model#导入训练模型文件
from models.loss import CollisionLoss, JointLimitLoss, RegLoss#导入损失函数
import dataset#导入数据库
from dataset import Normalize#导入归一化函数
from train import train_epoch#导入训练步骤
from test import test_epoch#导入测试步骤
from utils.config import cfg
from utils.util import create_folder#判断文件路径是否存在


import os#导入全局变量
import logging#导入加载器
import argparse#导入参数器
from datetime import datetime#导入时间

if __name__ == '__main__':
# Argument parse
    print("main.py is being loaded")
    # Argument parse
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--cfg', default='configs/train/arm.yaml', type=str, help='Path to configuration file')
    args = parser.parse_args()
    # Configurations parse
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    # print(cfg)
    # Create folder
    create_folder(cfg.OTHERS.SAVE)
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

    # Optional: 添加控制台输出
    # logger.addHandler(logging.StreamHandler())

    logger.info("Program started.coeff=100")
    logger.info("权值不变 oriloss不变")
    # logger.info("权值:ee=1000,vec=100,col=1000,lim=10000,ori=100,fin=100,reg=1,el=0")
    print("Handlers after setup:", logger.handlers)

    writer = SummaryWriter(os.path.join(cfg.OTHERS.SUMMARY, "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())))


    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # device = torch.device("cpu")
    # Load data
    pre_transform = transforms.Compose([Normalize()])
    train_set = getattr(dataset, cfg.DATASET.TRAIN.SOURCE_NAME)(root=cfg.DATASET.TRAIN.SOURCE_PATH, pre_transform=pre_transform)
    train_loader = DataListLoader(train_set, batch_size=cfg.HYPER.BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)

    # print(getattr(dataset, cfg.DATASET.TRAIN.TARGET_NAME))
    train_target = sorted([target for target in getattr(dataset, cfg.DATASET.TRAIN.TARGET_NAME)(root=cfg.DATASET.TRAIN.TARGET_PATH)], key=lambda target : target.skeleton_type)
    # print("-------------------------")
    # for batch_idx, data_list in enumerate(train_loader):
    #     for target_idx, target in enumerate(train_target):
    #         target_list = [target for data in data_list]
    #         print(target_list)
    # print("-------------------------")
    test_set = getattr(dataset, cfg.DATASET.TEST.SOURCE_NAME)(root=cfg.DATASET.TEST.SOURCE_PATH, pre_transform=pre_transform)

    test_loader = DataListLoader(test_set,batch_size=cfg.HYPER.BATCH_SIZE,shuffle=True,num_workers=1,pin_memory=True)
    # test_loader = DataListLoader(test_set, 
    #                              batch_size=cfg.HYPER.BATCH_SIZE, 
    #                              shuffle=False, 
    #                              num_workers=1, 
    #                              pin_memory=True)

    test_target = sorted([target for target in getattr(dataset, cfg.DATASET.TEST.TARGET_NAME)(root=cfg.DATASET.TEST.TARGET_PATH)], key=lambda target : target.skeleton_type)

    # Create model
    model = getattr(model, cfg.MODEL.NAME)().to(device)

    # Load checkpoint
    if cfg.MODEL.CHECKPOINT is not None:
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))
        print('Loaded checkpoint from {}'.format(cfg.MODEL.CHECKPOINT))

    # Create loss criterion
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
    fin_criterion =  None
    # fin_criterion = nn.MSELoss() if cfg.LOSS.FIN else None
    # regularization loss
    reg_criterion = RegLoss() if cfg.LOSS.REG else None
    # elbow loss
    # el_criterion = nn.MSELoss() if cfg.LOSS.EL else Nones
    el_criterion =  None
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.HYPER.LEARNING_RATE)

    best_loss = float('Inf')

    for epoch in range(cfg.HYPER.EPOCHS):
        # Start training
        train_loss = train_epoch(model, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, fin_criterion, reg_criterion,el_criterion,
                                 optimizer, train_loader, train_target, epoch, logger, cfg.OTHERS.LOG_INTERVAL, writer, device)

        # Start testing
        test_loss = test_epoch(model, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, fin_criterion, reg_criterion,el_criterion,
                               test_loader, test_target, epoch, logger, cfg.OTHERS.LOG_INTERVAL, writer, device)

        # Save model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), os.path.join(cfg.OTHERS.SAVE, "best_model_epoch_{:04d}.pth".format(epoch)))
            logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '-'))
