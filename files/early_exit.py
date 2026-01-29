"""
Early Exit Mechanism for CE-CoLLM
实现论文中的延迟感知早退机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class EarlyExitHead(nn.Module):
    """
    早退预测头
    在中间层添加一个轻量级的分类器来生成token预测
    """
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        # 简单的两层MLP作为早退头
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        x = self.layer_norm(hidden_states)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


class LatencyAwareEarlyExit:
    """
    延迟感知的早退机制
    根据置信度决定是否在当前层退出
    """
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self.exit_statistics = {
            'total_tokens': 0,
            'early_exit_tokens': 0,
            'cloud_request_tokens': 0,
            'exit_layer_counts': {}
        }
    
    def compute_confidence(self, logits: torch.Tensor) -> Tuple[float, int]:
        """
        计算预测的置信度
        
        Args:
            logits: [vocab_size] 或 [batch_size, vocab_size]
        
        Returns:
            confidence: 最大概率值
            predicted_token: 预测的token ID
        """
        if logits.dim() > 1:
            logits = logits[-1]  # 取最后一个位置的logits
        
        probs = F.softmax(logits, dim=-1)
        confidence, predicted_token = torch.max(probs, dim=-1)
        
        return confidence.item(), predicted_token.item()
    
    def should_exit(
        self, 
        logits: torch.Tensor, 
        layer_idx: int,
        is_last_edge_layer: bool = False
    ) -> Tuple[bool, Optional[int], float]:
        """
        判断是否应该在当前层退出
        
        Args:
            logits: 当前层的输出logits
            layer_idx: 当前层的索引
            is_last_edge_layer: 是否是边缘分区的最后一层
        
        Returns:
            should_exit: 是否应该退出
            token_id: 如果退出，返回预测的token ID
            confidence: 置信度分数
        """
        confidence, token_id = self.compute_confidence(logits)
        
        # 更新统计信息
        self.exit_statistics['total_tokens'] += 1
        
        # 如果置信度超过阈值，退出
        if confidence >= self.confidence_threshold:
            self.exit_statistics['early_exit_tokens'] += 1
            if layer_idx not in self.exit_statistics['exit_layer_counts']:
                self.exit_statistics['exit_layer_counts'][layer_idx] = 0
            self.exit_statistics['exit_layer_counts'][layer_idx] += 1
            return True, token_id, confidence
        
        # 如果是边缘分区的最后一层但置信度不足，需要请求云支持
        if is_last_edge_layer:
            self.exit_statistics['cloud_request_tokens'] += 1
            return False, None, confidence
        
        # 继续到下一层
        return False, None, confidence
    
    def get_statistics(self) -> dict:
        """获取早退统计信息"""
        stats = self.exit_statistics.copy()
        if stats['total_tokens'] > 0:
            stats['early_exit_rate'] = stats['early_exit_tokens'] / stats['total_tokens']
            stats['cloud_request_rate'] = stats['cloud_request_tokens'] / stats['total_tokens']
        else:
            stats['early_exit_rate'] = 0.0
            stats['cloud_request_rate'] = 0.0
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.exit_statistics = {
            'total_tokens': 0,
            'early_exit_tokens': 0,
            'cloud_request_tokens': 0,
            'exit_layer_counts': {}
        }


class EarlyExitLLM(nn.Module):
    """
    带有早退机制的LLM模型
    在指定的层添加早退头
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        early_exit_layers: list,
        hidden_size: int,
        vocab_size: int
    ):
        super().__init__()
        self.base_model = base_model
        self.early_exit_layers = sorted(early_exit_layers)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # 为每个早退层创建早退头
        self.exit_heads = nn.ModuleDict({
            str(layer_idx): EarlyExitHead(hidden_size, vocab_size)
            for layer_idx in early_exit_layers
        })
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        return_hidden_states: bool = False,
        max_layer: Optional[int] = None
    ) -> dict:
        """
        前向传播，支持早退
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            past_key_values: KV缓存
            return_hidden_states: 是否返回所有隐藏状态
            max_layer: 最大执行到哪一层（用于边缘分区）
        
        Returns:
            outputs: 包含logits, hidden_states, past_key_values等
        """
        outputs = {
            'exit_layer': None,
            'exit_logits': None,
            'all_exit_logits': {},
            'hidden_states': None,
            'past_key_values': None
        }
        
        # 这里简化实现，实际应该调用base_model的各层
        # 在实际实现中需要访问transformer的各层
        hidden_states = None
        
        # 遍历各层（简化版本）
        for layer_idx in range(max_layer if max_layer else len(self.base_model.layers)):
            # 执行transformer层
            # hidden_states = self.base_model.layers[layer_idx](hidden_states, ...)
            
            # 如果当前层有早退头，计算早退logits
            if layer_idx in self.early_exit_layers:
                exit_logits = self.exit_heads[str(layer_idx)](hidden_states)
                outputs['all_exit_logits'][layer_idx] = exit_logits
        
        outputs['hidden_states'] = hidden_states
        return outputs


def train_early_exit_heads(
    model: EarlyExitLLM,
    dataloader,
    optimizer,
    num_epochs: int = 3,
    device: str = 'cuda'
):
    """
    训练早退头
    使用蒸馏损失让早退头学习最后一层的输出
    
    Args:
        model: 带早退的模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 设备
    """
    model.train()
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            # 获取最后一层的输出作为teacher
            with torch.no_grad():
                teacher_outputs = model.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
            
            # 获取早退头的输出
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_hidden_states=True
            )
            
            # 计算所有早退头的蒸馏损失
            loss = 0
            for layer_idx, exit_logits in outputs['all_exit_logits'].items():
                student_log_probs = F.log_softmax(exit_logits, dim=-1)
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                loss += criterion(student_log_probs, teacher_probs)
            
            loss = loss / len(outputs['all_exit_logits'])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
