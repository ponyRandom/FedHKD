import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics

# 添加DOT算法的导入
from flcore.optimizers.dot import DistillationOrientedTrainer

def _get_gt_mask(logits, target):
    """获取目标类别的mask"""
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    """获取非目标类别的mask"""
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    """将目标类别和非目标类别的概率连接"""
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    """
    解耦知识蒸馏损失函数（数值稳定版本）
    Args:
        logits_student: 学生模型的logits
        logits_teacher: 教师模型的logits
        target: 真实标签
        alpha: TCKD的权重
        beta: NCKD的权重
        temperature: 温度参数
    """
    # 添加数值稳定性检查
    if torch.isnan(logits_student).any() or torch.isnan(logits_teacher).any():
        print("Warning: NaN detected in logits")
        return torch.tensor(0.0, device=logits_student.device, requires_grad=True)

    if torch.isinf(logits_student).any() or torch.isinf(logits_teacher).any():
        print("Warning: Inf detected in logits")
        return torch.tensor(0.0, device=logits_student.device, requires_grad=True)

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    # 使用更稳定的softmax计算
    pred_student_logits = F.softmax(logits_student / temperature, dim=1)
    pred_teacher_logits = F.softmax(logits_teacher / temperature, dim=1)
    # These cat_mask outputs are (Batch, 2)
    pred_student_cat = cat_mask(pred_student_logits, gt_mask, other_mask)
    pred_teacher_cat = cat_mask(pred_teacher_logits, gt_mask, other_mask)

    # 添加小的epsilon避免log(0)
    epsilon = 1e-8
    # For TCKD, log_pred_student should be log of the (Batch, 2) tensor
    log_pred_student_cat = torch.log(pred_student_cat + epsilon)

    # TCKD loss (Target Class Knowledge Distillation)
    tckd_loss = (
        F.kl_div(log_pred_student_cat, pred_teacher_cat, reduction='sum') # Use (B,2) inputs and reduction='sum'
        # * (temperature**2)
        / target.shape[0]
    )

    # NCKD loss (Non-Target Class Knowledge Distillation) - 使用更稳定的计算方式
    # 不使用大数值mask，而是直接计算非目标类别的分布
    logits_teacher_masked = logits_teacher.clone()
    logits_student_masked = logits_student.clone()

    # 将目标类别的logits设置为很小的值而不是-1000
    logits_teacher_masked = logits_teacher_masked.masked_fill(gt_mask, -1e2)
    logits_student_masked = logits_student_masked.masked_fill(gt_mask, -1e2)

    pred_teacher_part2 = F.softmax(logits_teacher_masked / temperature, dim=1)
    log_pred_student_part2 = F.log_softmax(logits_student_masked / temperature, dim=1)

    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum') # Use reduction='sum'
        # * (temperature**2)
        / target.shape[0]
    )

    # 检查损失是否为NaN
    total_loss = alpha * tckd_loss + beta * nckd_loss
    if torch.isnan(total_loss):
        print("Warning: NaN in DKD loss, returning zero")
        return torch.tensor(0.0, device=logits_student.device, requires_grad=True)

    return total_loss

class clientHKD(Client):
    # def __init__(self, args, id, train_samples, test_samples, **kwargs): #全局本地相同架构
    #     super().__init__(args, id, train_samples, test_samples, **kwargs)

    #     self.alpha = args.alpha
    #     self.beta = args.beta
    #     self.temperature = args.temperature
        
    #     # Add warmup related fields
    #     self.warmup_rounds = args.warmup_rounds if hasattr(args, 'warmup_rounds') else 10
    #     self.current_round = 0  # Will be set by server in each round
        
    #     self.mentee_learning_rate = args.mentee_learning_rate

    #     self.global_model = copy.deepcopy(args.model)
    #     self.optimizer_g = torch.optim.SGD(self.global_model.parameters(), lr=self.mentee_learning_rate)
    #     self.learning_rate_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
    #         optimizer=self.optimizer_g, 
    #         gamma=args.learning_rate_decay_gamma
    #     )

    #     self.feature_dim = list(args.model.head.parameters())[0].shape[1]
    #     self.W_h = nn.Linear(self.feature_dim, self.feature_dim, bias=False).to(self.device)
    #     self.optimizer_W = torch.optim.SGD(self.W_h.parameters(), lr=self.learning_rate)
    #     self.learning_rate_scheduler_W = torch.optim.lr_scheduler.ExponentialLR(
    #         optimizer=self.optimizer_W, 
    #         gamma=args.learning_rate_decay_gamma
    #     )

    #     self.KL = nn.KLDivLoss()
    #     self.MSE = nn.MSELoss()
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.alpha = args.alpha
        self.beta = args.beta
        self.temperature = args.temperature
        
        # Add warmup related fields
        self.warmup_rounds = args.warmup_rounds if hasattr(args, 'warmup_rounds') else 10
        self.current_round = 0
        
        self.mentee_learning_rate = args.mentee_learning_rate

        # 使用传入的全局模型（可能与本地模型架构不同）
        if hasattr(args, 'global_model'):
            self.global_model = copy.deepcopy(args.global_model)
        else:
            # 后备方案：使用与本地模型相同的架构
            self.global_model = copy.deepcopy(args.model)

        # DOT算法相关参数 - 调整为更稳定的值
        self.momentum = args.hkd_momentum
        self.delta = args.delta
        self.momentum_kd = self.momentum + self.delta
        self.momentum_task = max(0.1, self.momentum - self.delta)  # 确保momentum_task不会太小
        print(f"momentum_kd: {self.momentum_kd}, momentum_task: {self.momentum_task}")
        
        # 使用DOT优化器替换原有的SGD优化器
        # 本地模型的DOT优化器
        self.optimizer = DistillationOrientedTrainer(
            params=self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum_task,
            momentum_kd=self.momentum_kd,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 1e-4
        )
        
        # 全局模型的DOT优化器
        self.optimizer_g = DistillationOrientedTrainer(
            params=self.global_model.parameters(),
            lr=self.mentee_learning_rate,
            momentum=self.momentum_task,
            momentum_kd=self.momentum_kd,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 1e-4
        )
        
        # 保留学习率调度器（需要适配DOT优化器）
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_g, 
            gamma=args.learning_rate_decay_gamma
        )

        # 获取特征维度 - 需要处理不同架构的情况
        local_feature_dim = list(args.model.head.parameters())[0].shape[1]
        global_feature_dim = list(self.global_model.head.parameters())[0].shape[1]
        
        # 创建特征对齐层
        self.feature_align = nn.Linear(global_feature_dim, local_feature_dim, bias=False).to(self.device)
        self.optimizer_align = torch.optim.SGD(self.feature_align.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_align = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_align, 
            gamma=args.learning_rate_decay_gamma
        )
        
        # W_h 特征变换矩阵
        self.W_h = nn.Linear(local_feature_dim, local_feature_dim, bias=False).to(self.device)
        self.optimizer_W = torch.optim.SGD(self.W_h.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_W = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_W, 
            gamma=args.learning_rate_decay_gamma
        )

        self.KL = nn.KLDivLoss()
        self.MSE = nn.MSELoss()

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        self.global_model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # 修改warmup机制，使其更平滑且持续更长时间
        warmup_factor = min(1, self.current_round / (self.warmup_rounds * 2)) if self.warmup_rounds > 0 else 0.5
        
        # 添加动态权重调整，防止过度蒸馏
        kd_weight = 0.1 + 0.4 * warmup_factor  # 知识蒸馏权重从0.1逐渐增加到0.5

        torch.cuda.empty_cache()

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                # 前向传播
                rep = self.model.base(x)
                rep_g = self.global_model.base(x)
                
                # 特征对齐
                if self.feature_align is not None:
                    rep_g_aligned = self.feature_align(rep_g)
                else:
                    rep_g_aligned = rep_g
                
                output = self.model.head(rep)
                output_g = self.global_model.head(rep_g)

                # === DOT算法：第一阶段 - 处理知识蒸馏损失 ===
                # 计算知识蒸馏损失，使用动态权重
                L_d = kd_weight * dkd_loss(output, output_g, y, self.alpha, self.beta, self.temperature)
                L_d_g = kd_weight * dkd_loss(output_g, output, y, self.alpha, self.beta, self.temperature)
                
                # 计算CE损失用于特征损失的归一化
                CE_loss_temp = self.loss(output, y)
                CE_loss_g_temp = self.loss(output_g, y)
                denominator = CE_loss_temp + CE_loss_g_temp
                
                # 降低特征损失的权重，防止过度对齐
                # feature_weight = 0.05 * kd_weight
                feature_weight = kd_weight
                L_h = feature_weight * self.MSE(rep, self.W_h(rep_g_aligned)) / (denominator + 1e-8)
                L_h_g = feature_weight * self.MSE(rep, self.W_h(rep_g_aligned)) / (denominator + 1e-8)

                # 知识蒸馏损失总和
                kd_loss = L_d + L_h
                kd_loss_g = L_d_g + L_h_g

                # 清零梯度
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_g.zero_grad(set_to_none=True)
                self.optimizer_W.zero_grad()
                if self.feature_align is not None:
                    self.optimizer_align.zero_grad()

                # 知识蒸馏损失反向传播
                if kd_loss.item() > 1e-8:  # 只有当KD损失足够大时才反向传播
                    kd_loss.backward(retain_graph=True)
                if kd_loss_g.item() > 1e-8:
                    kd_loss_g.backward(retain_graph=True)

                # 梯度裁剪 - 使用更保守的裁剪把原来的10改成5
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(self.W_h.parameters(), 5)
                if self.feature_align is not None:
                    torch.nn.utils.clip_grad_norm_(self.feature_align.parameters(), 5)

                # DOT: 保存知识蒸馏梯度
                self.optimizer.step_kd()
                self.optimizer_g.step_kd()
                
                # 更新辅助网络（W_h和feature_align）
                self.optimizer_W.step()
                if self.feature_align is not None:
                    self.optimizer_align.step()

                # === DOT算法：第二阶段 - 处理任务损失 ===
                # 清零梯度
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_g.zero_grad(set_to_none=True)

                # 重新计算输出（避免梯度污染）
                rep = self.model.base(x)
                rep_g = self.global_model.base(x)
                output = self.model.head(rep)
                output_g = self.global_model.head(rep_g)

                # 计算任务损失
                CE_loss = self.loss(output, y)
                CE_loss_g = self.loss(output_g, y)

                # 任务损失反向传播
                CE_loss.backward()
                CE_loss_g.backward()

                # 梯度裁剪把10改成5
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 5)

                # DOT: 处理任务损失梯度并更新参数
                self.optimizer.step()
                self.optimizer_g.step()

        self.compressed_param = {name: param.cpu().detach().numpy() for name, param in
                                self.global_model.named_parameters()}
        
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_g.step()
            self.learning_rate_scheduler_W.step()
            if self.feature_align is not None:
                self.learning_rate_scheduler_align.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        
    def set_parameters(self, global_param, energy):
        # recover
        # for k in global_param.keys():
        #     if len(global_param[k]) == 3:
        #         # use np.matmul to support high-dimensional CNN param
        #         global_param[k] = np.matmul(global_param[k][0] * global_param[k][1][..., None, :], global_param[k][2])
        
        for name, old_param in self.global_model.named_parameters():
            if name in global_param:
                old_param.data = torch.tensor(global_param[name], device=self.device).data.clone()
        # self.energy = energy
    
    # def train_metrics(self): #这是clientkd的方法
    #     trainloader = self.load_train_data()
    #     # self.model = self.load_model('model')
    #     # self.model.to(self.device)
    #     self.model.eval()
    #     self.global_model.eval()

    #     train_num = 0
    #     losses = 0
    #     with torch.no_grad():
    #         for x, y in trainloader:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             rep = self.model.base(x)
    #             rep_g = self.global_model.base(x)
    #             output = self.model.head(rep)
    #             output_g = self.global_model.head(rep_g)

    #             CE_loss = self.loss(output, y)
    #             CE_loss_g = self.loss(output_g, y)
    #             L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) / (CE_loss + CE_loss_g)
    #             L_h = self.MSE(rep, self.W_h(rep_g)) / (CE_loss + CE_loss_g)

    #             loss = CE_loss + L_d + L_h
    #             train_num += y.shape[0]
    #             losses += loss.item() * y.shape[0]

    #     # self.model.cpu()
    #     # self.save_model(self.model, 'model')

    #     return losses, train_num

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()
        self.global_model.eval()
    
        train_num = 0
        total_loss_val = 0
        total_ce_loss_val = 0
        total_l_d_val = 0
        total_l_h_val = 0
    
        # 使用与训练时相同的动态权重
        warmup_factor = min(1, self.current_round / (self.warmup_rounds * 2)) if self.warmup_rounds > 0 else 0.5
        kd_weight = 0.1 + 0.4 * warmup_factor
        feature_weight = kd_weight
    
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
    
                # 添加try-except块来捕获和处理可能的数值问题
                try:
                    rep = self.model.base(x)
                    rep_g = self.global_model.base(x)
    
                    # 检查并处理可能的NaN值
                    if torch.isnan(rep).any() or torch.isnan(rep_g).any():
                        print("Warning: NaN detected in representations, skipping batch")
                        continue

                    # 如果特征维度不同，需要对齐
                    if self.feature_align is not None:
                        rep_g_aligned = self.feature_align(rep_g)
                    else:
                        rep_g_aligned = rep_g
    
                    output = self.model.head(rep)
                    output_g = self.global_model.head(rep_g)
    
                    # 检查输出是否包含NaN
                    if torch.isnan(output).any() or torch.isnan(output_g).any():
                        print("Warning: NaN detected in outputs, skipping batch")
                        continue
    
                    ce_loss = self.loss(output, y)
                    ce_loss_g_for_den = self.loss(output_g, y)
    
                    denominator = ce_loss + ce_loss_g_for_den
    
                    if denominator.item() < 1e-8:
                        l_d = torch.tensor(0.0, device=self.device)
                        l_h = torch.tensor(0.0, device=self.device)
                    else:
                        l_d = kd_weight * dkd_loss(output, output_g, y, self.alpha, self.beta, self.temperature)
                        l_h = feature_weight * self.MSE(rep, self.W_h(rep_g_aligned)) / denominator
    
                    # 检查损失值是否为NaN
                    if torch.isnan(l_d).any() or torch.isnan(l_h).any() or torch.isnan(ce_loss).any():
                        print("Warning: NaN detected in loss computation, skipping batch")
                        continue
    
                    current_batch_loss = ce_loss + l_d + l_h
    
                    train_num += y.shape[0]
                    total_loss_val += current_batch_loss.item() * y.shape[0]
                    total_ce_loss_val += ce_loss.item() * y.shape[0]
                    total_l_d_val += l_d.item() * y.shape[0]
                    total_l_h_val += l_h.item() * y.shape[0]
    
                except RuntimeError as e:
                    print(f"Warning: Runtime error in batch processing: {str(e)}")
                    continue
    
        metrics = {
            'loss': total_loss_val / max(train_num, 1),  # 避免除以0
            'ce_loss': total_ce_loss_val / max(train_num, 1),
            'l_d': total_l_d_val / max(train_num, 1),
            'l_h': total_l_h_val / max(train_num, 1),
            'num_samples': train_num
        }
        return metrics
    
    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()
    
        test_acc = 0
        test_loss = 0
        test_num = 0
        y_prob = []
        y_true = []
    
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
    
                # 计算交叉熵损失
                loss = self.loss(output, y)
                test_loss += loss.item() * y.shape[0]
    
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
    
                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
    
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
    
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
    
        return test_acc, test_loss, test_num, auc