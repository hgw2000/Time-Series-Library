import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding  # 数据嵌入层
from layers.Conv_Blocks import Inception_Block_V1  # Inception卷积块


def FFT_for_Period(x, k=2):
    """
    通过FFT（快速傅里叶变换）检测时间序列中的主要周期性模式
    输入:
        x: 形状为[B, T, C]的张量，其中B为批次大小，T为序列长度，C为特征维度
        k: 需要检测的主要周期数量
    返回:
        period: 检测到的周期长度列表
        period_weight: 每个周期对应的重要性权重
    """
    xf = torch.fft.rfft(x, dim=1)  # 对时间维度进行FFT变换
    frequency_list = abs(xf).mean(0).mean(-1)  # 计算频率振幅的平均值
    frequency_list[0] = 0  # 忽略零频率分量（直流分量）
    _, top_list = torch.topk(frequency_list, k)  # 选择k个最显著的频率
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list  # 将频率转换为对应的周期长度
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    """
    TimesNet的核心模块：时间-空间混合块
    通过FFT检测周期性，并使用2D卷积捕捉时间和特征维度的相关性
    """
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len  # 输入序列长度
        self.pred_len = configs.pred_len  # 预测序列长度
        self.k = configs.top_k  # 要检测的周期数量
        
        # 使用Inception块构建参数高效的卷积网络
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                             num_kernels=configs.num_kernels),  # 第一层Inception块
            nn.GELU(),  # 非线性激活函数
            Inception_Block_V1(configs.d_ff, configs.d_model,
                             num_kernels=configs.num_kernels)  # 第二层Inception块
        )

    def forward(self, x):
        """
        前向传播过程
        x: 输入张量 [批次大小, 时间步长, 特征维度]
        """
        B, T, N = x.size()
        
        # 1. 周期检测
        period_list, period_weight = FFT_for_Period(x, self.k)
        
        res = []
        # 2. 对每个检测到的周期进行处理
        for i in range(self.k):
            period = period_list[i]
            
            # 2.1 序列长度填充，确保可以被周期整除
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([B, length - (self.seq_len + self.pred_len), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            
            # 2.2 重排张量为周期性2D结构
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            
            # 2.3 使用2D卷积处理重排后的数据
            out = self.conv(out)
            
            # 2.4 恢复原始维度顺序
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        
        # 3. 合并不同周期的结果
        res = torch.stack(res, dim=-1)
        
        # 4. 基于周期重要性进行加权聚合
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        # 5. 残差连接
        res = res + x
        return res


class Model(nn.Module):
    """
    TimesNet完整模型实现
    支持多种时序任务：
    - 长期/短期预测
    - 缺失值填补
    - 异常检测
    - 时序分类
    论文链接：https://openreview.net/pdf?id=ju_Uqw384Oq
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name  # 任务类型
        self.seq_len = configs.seq_len      # 输入序列长度
        self.label_len = configs.label_len  # 标签长度
        self.pred_len = configs.pred_len    # 预测长度
        
        # 构建模型组件
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, 
                                         configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        # 根据任务类型配置输出层
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        时序预测任务的前向传播
        包含：数据标准化 -> 特征提取 -> 序列预测 -> 反标准化
        """
        # 1. 数据标准化（来自Non-stationary Transformer）
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 2. 特征提取和预测
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # 数据嵌入
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # 时间维度对齐
        
        # 3. 通过TimesNet层
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        # 4. 输出投影
        dec_out = self.projection(enc_out)

        # 5. 反标准化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        模型统一入口，根据任务类型调用相应的处理方法
        """
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # 只返回预测部分
        elif self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        elif self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        elif self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
