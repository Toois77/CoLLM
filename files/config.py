"""
CE-CoLLM Configuration
配置文件定义了云边协作LLM系统的所有关键参数
"""

import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class CECoLLMConfig:
    """CE-CoLLM框架配置"""
    
    # 模型配置
    model_name: str = "meta-llama/Llama-2-7b-hf"
    num_layers: int = 32  # LLaMA-7B有32层
    hidden_size: int = 4096
    vocab_size: int = 32000
    
    # 早退配置
    early_exit_layers: list = None  # 早退层位置，例如 [8, 16]
    confidence_threshold: float = 0.8  # 置信度阈值
    
    # 云边分割配置
    edge_num_layers: int = 16  # 边缘设备上的层数
    cloud_num_layers: int = 16  # 云端的层数
    
    # 通信配置
    use_fp16_transfer: bool = True  # 使用FP16传输以减少数据量
    async_upload: bool = True  # 异步上传上下文
    upload_trigger_layer: int = 8  # 触发异步上传的层
    
    # 云端API配置
    cloud_server_url: str = "http://localhost:8000"
    context_endpoint: str = "/context/upload"
    inference_endpoint: str = "/inference/continue"
    
    # 推理配置
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    
    # 性能配置
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 会话管理
    session_timeout: int = 3600  # 1小时超时
    
    # 运行模式
    mode: str = "collaborative"  # "standalone" 或 "collaborative"
    
    def __post_init__(self):
        if self.early_exit_layers is None:
            # 默认在边缘分区的中间和末尾设置早退点
            self.early_exit_layers = [
                self.edge_num_layers // 2,  # 中间层
                self.edge_num_layers  # 最后一层
            ]
        
        assert self.edge_num_layers + self.cloud_num_layers == self.num_layers, \
            "边缘层数和云端层数之和必须等于总层数"
        
        assert all(layer <= self.edge_num_layers for layer in self.early_exit_layers), \
            "早退层必须在边缘分区内"


# 默认配置实例
default_config = CECoLLMConfig()
