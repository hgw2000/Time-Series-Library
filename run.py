"""
时间序列分析主入口脚本
功能：
1. 参数解析与实验配置：支持5种时间序列任务类型
2. 设备初始化与随机种子设置：确保实验可复现性
3. 任务分发机制：根据任务类型初始化对应实验模块
4. 训练流程控制：包含模型保存和GPU内存管理
"""
import argparse
import os
import torch
import torch.backends
# 导入不同任务的实验类
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast  # 长期预测任务
from exp.exp_imputation import Exp_Imputation                     # 数据补全任务
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast # 短期预测任务
from exp.exp_anomaly_detection import Exp_Anomaly_Detection       # 异常检测任务
from exp.exp_classification import Exp_Classification             # 分类任务
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    # 固定随机种子保证实验可复现
    fix_seed = 2021  
    random.seed(fix_seed)        # Python随机种子
    torch.manual_seed(fix_seed)  # PyTorch随机种子
    np.random.seed(fix_seed)     # Numpy随机种子

    # 创建参数解析器
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True, 
                        help='是否使用GPU，默认为True')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='使用的GPU编号，默认为0')
    parser.add_argument('--gpu_type', type=str, default='cuda', 
                        help='GPU类型选项：[cuda, mps]，默认为cuda')
    parser.add_argument('--use_multi_gpu', action='store_true', 
                        help='是否使用多GPU并行训练，添加此参数表示启用', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='多GPU使用的设备ID列表，用逗号分隔')

    # 非平稳投影器参数
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='投影器隐藏层维度列表，例如：[128, 128]表示两层128维')
    parser.add_argument('--p_hidden_layers', type=int, default=2,
                        help='投影器的隐藏层层数，默认为2层')

    # 评估指标
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='是否使用DTW指标，默认为False（DTW计算耗时，非必要不建议开启）')

    # 数据增强配置
    parser.add_argument('--augmentation_ratio', type=int, default=0, 
                        help="数据增强倍数，0表示不增强")
    parser.add_argument('--seed', type=int, default=2, 
                        help="数据增强的随机种子，默认为2")
    # 具体增强方法开关
    parser.add_argument('--jitter', default=False, action="store_true", 
                        help="启用抖动增强")
    parser.add_argument('--scaling', default=False, action="store_true",
                        help="启用缩放增强")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="等长排列增强")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="随机长度排列增强")
    parser.add_argument('--magwarp', default=False, action="store_true",
                        help="幅度弯曲增强")
    parser.add_argument('--timewarp', default=False, action="store_true",
                        help="时间弯曲增强")
    parser.add_argument('--windowslice', default=False, action="store_true",
                        help="窗口切片增强")
    parser.add_argument('--windowwarp', default=False, action="store_true",
                        help="窗口变形增强")
    parser.add_argument('--rotation', default=False, action="store_true",
                        help="旋转增强")
    parser.add_argument('--spawner', default=False, action="store_true",
                        help="生成器增强")
    parser.add_argument('--dtwwarp', default=False, action="store_true",
                        help="DTW弯曲增强")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true",
                        help="形状DTW增强")
    parser.add_argument('--wdba', default=False, action="store_true",
                        help="加权DBA增强")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="判别式DTW增强")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="判别式形状DTW增强")
    parser.add_argument('--extra_tag', type=str, default="",
                        help="额外标签，用于实验备注")

    # TimeXer模型参数
    parser.add_argument('--patch_len', type=int, default=16, 
                        help='TimeXer模型的补丁长度，默认为16')

    # 解析所有参数
    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    # 任务分发逻辑
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast       # 长期预测实验
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast      # 短期预测实验
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation               # 数据补全实验
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection        # 异常检测实验
    elif args.task_name == 'classification':
        Exp = Exp_Classification           # 分类实验
    else:
        Exp = Exp_Long_Term_Forecast       # 默认使用长期预测

    # 训练模式
    if args.is_training:
        # 多实验循环（itr控制实验次数）
        for ii in range(args.itr):
            # 实验初始化
            exp = Exp(args)  # 创建对应任务的实验对象
            
            # 生成实验设置名称（包含所有关键参数）
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,    # 任务类型
                args.model_id,     # 模型ID
                args.model,        # 模型名称
                args.data,         # 数据集
                args.features,     # 特征类型
                args.seq_len,      # 输入序列长度
                args.label_len,    # 标签长度
                args.pred_len,     # 预测长度
                args.d_model,      # 模型维度
                args.n_heads,      # 注意力头数
                args.e_layers,     # 编码器层数
                args.d_layers,     # 解码器层数
                args.d_ff,         # 前馈网络维度
                args.expand,       # Mamba扩展因子
                args.d_conv,       # Mamba卷积核尺寸
                args.factor,       # 注意力因子
                args.embed,        # 时间编码方式
                args.distil,       # 是否使用蒸馏
                args.des, ii)      # 实验描述和迭代次数

            # 训练阶段
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)     # 执行训练流程

            # 测试阶段
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)      # 执行测试流程
            
            # GPU缓存清理
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()  # MPS设备缓存清理
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()          # CUDA设备缓存清理
    
    # 仅测试模式
    else:
        exp = Exp(args)  # 初始化实验对象
        ii = 0
        # 生成测试设置名称（格式与训练相同）
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()