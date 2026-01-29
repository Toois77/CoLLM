"""
Edge Inference Engine for CE-CoLLM
边缘端推理引擎，负责本地推理和云端协作
"""

import torch
import asyncio
import aiohttp
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

from config import CECoLLMConfig
from early_exit import LatencyAwareEarlyExit, EarlyExitHead


@dataclass
class InferenceContext:
    """推理上下文数据"""
    session_id: str
    hidden_states: torch.Tensor
    past_key_values: Optional[tuple]
    attention_mask: torch.Tensor
    input_ids: List[int]
    generated_tokens: List[int]
    

class EdgeInferenceEngine:
    """
    边缘推理引擎
    实现CE-CoLLM的边缘侧推理逻辑
    """
    
    def __init__(self, config: CECoLLMConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 加载分词器
        print(f"Loading tokenizer: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载边缘端模型（前N层）
        print(f"Loading edge model ({config.edge_num_layers} layers)...")
        self.edge_model = self._load_edge_model()
        
        # 创建早退头
        self.exit_heads = torch.nn.ModuleDict({
            str(layer): EarlyExitHead(config.hidden_size, config.vocab_size)
            for layer in config.early_exit_layers
        })
        self.exit_heads.to(self.device)
        
        # 早退机制
        self.early_exit = LatencyAwareEarlyExit(config.confidence_threshold)
        
        # 会话管理
        self.sessions: Dict[str, InferenceContext] = {}
        
        # 性能统计
        self.performance_stats = {
            'edge_inference_time': [],
            'cloud_inference_time': [],
            'communication_time': [],
            'async_upload_time': [],
            'total_inference_time': []
        }
    
    def _load_edge_model(self) -> torch.nn.Module:
        """
        加载边缘端模型分区
        实际实现中需要只加载前N层
        这里简化处理
        """
        # 加载完整模型
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.use_fp16_transfer else torch.float32,
            device_map=self.device
        )
        
        # 在实际实现中，这里应该只保留前edge_num_layers层
        # 这需要对模型架构有深入了解
        # 简化版本：使用完整模型，但在推理时只执行前N层
        
        model.eval()
        return model
    
    async def async_upload_context(
        self,
        session_id: str,
        hidden_states: torch.Tensor,
        past_key_values: Optional[tuple] = None
    ):
        """
        异步上传上下文数据到云端
        
        Args:
            session_id: 会话ID
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
            past_key_values: KV缓存
        """
        upload_start = time.time()
        
        try:
            # 转换为FP16以减少传输量
            if self.config.use_fp16_transfer:
                hidden_states = hidden_states.half()
            
            # 准备上传数据
            context_data = {
                'session_id': session_id,
                'hidden_states': hidden_states.cpu().numpy().tobytes(),
                'hidden_states_shape': list(hidden_states.shape),
                'dtype': 'float16' if self.config.use_fp16_transfer else 'float32'
            }
            
            # 如果有KV缓存，也上传（简化处理）
            if past_key_values is not None:
                # 序列化KV缓存
                context_data['has_kv_cache'] = True
            
            # 异步HTTP请求
            url = f"{self.config.cloud_server_url}{self.config.context_endpoint}"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=context_data) as response:
                    if response.status == 200:
                        upload_time = time.time() - upload_start
                        self.performance_stats['async_upload_time'].append(upload_time)
                        print(f"✓ Context uploaded in {upload_time:.3f}s")
                    else:
                        print(f"✗ Context upload failed: {response.status}")
        
        except Exception as e:
            print(f"✗ Context upload error: {e}")
    
    async def request_cloud_inference(
        self,
        session_id: str,
        current_input_ids: List[int]
    ) -> Optional[int]:
        """
        请求云端继续推理
        
        Args:
            session_id: 会话ID
            current_input_ids: 当前输入token序列
        
        Returns:
            next_token_id: 云端生成的下一个token
        """
        cloud_start = time.time()
        
        try:
            request_data = {
                'session_id': session_id,
                'input_ids': current_input_ids,
                'temperature': self.config.temperature,
                'top_p': self.config.top_p
            }
            
            url = f"{self.config.cloud_server_url}{self.config.inference_endpoint}"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=request_data, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        next_token = result.get('next_token')
                        
                        cloud_time = time.time() - cloud_start
                        self.performance_stats['cloud_inference_time'].append(cloud_time)
                        
                        return next_token
                    else:
                        print(f"✗ Cloud inference failed: {response.status}")
                        return None
        
        except Exception as e:
            print(f"✗ Cloud inference error: {e}")
            return None
    
    def edge_forward_with_early_exit(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[tuple] = None
    ) -> Tuple[Optional[int], torch.Tensor, tuple, bool]:
        """
        边缘端前向传播，支持早退
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            past_key_values: KV缓存
        
        Returns:
            token_id: 如果早退成功，返回token ID
            hidden_states: 当前隐藏状态
            past_key_values: 更新后的KV缓存
            need_cloud: 是否需要云端支持
        """
        with torch.no_grad():
            # 执行边缘模型的前向传播
            # 注意：这里需要逐层执行以支持早退
            # 简化版本使用整个模型
            outputs = self.edge_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True
            )
            
            hidden_states_list = outputs.hidden_states
            past_key_values = outputs.past_key_values
            
            # 在早退层检查置信度
            for layer_idx in self.config.early_exit_layers:
                if layer_idx >= len(hidden_states_list):
                    continue
                
                # 获取该层的隐藏状态
                layer_hidden_states = hidden_states_list[layer_idx]
                
                # 使用早退头预测
                exit_logits = self.exit_heads[str(layer_idx)](layer_hidden_states)
                
                # 判断是否应该退出
                is_last_layer = (layer_idx == self.config.early_exit_layers[-1])
                should_exit, token_id, confidence = self.early_exit.should_exit(
                    exit_logits[:, -1, :],  # 只看最后一个位置
                    layer_idx,
                    is_last_edge_layer=is_last_layer
                )
                
                if should_exit:
                    print(f"  ✓ Early exit at layer {layer_idx} (confidence: {confidence:.3f})")
                    return token_id, layer_hidden_states, past_key_values, False
                
                if is_last_layer:
                    # 最后一层也无法满足置信度，需要云端支持
                    print(f"  ☁ Need cloud support (confidence: {confidence:.3f})")
                    return None, layer_hidden_states, past_key_values, True
            
            # 不应该到达这里
            return None, hidden_states_list[-1], past_key_values, True
    
    async def generate_standalone(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        独立边缘推理模式（低延迟模式）
        完全在边缘设备上生成，不依赖云端
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
        
        Returns:
            生成的文本
        """
        print("\n=== Standalone Edge Inference Mode ===")
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        generated_tokens = []
        past_key_values = None
        
        inference_start = time.time()
        
        for step in range(max_new_tokens):
            # 边缘前向传播
            token_id, hidden_states, past_key_values, need_cloud = \
                self.edge_forward_with_early_exit(
                    input_ids,
                    attention_mask,
                    past_key_values
                )
            
            if token_id is None:
                # 在独立模式下，即使置信度不足也强制选择最可能的token
                with torch.no_grad():
                    exit_logits = self.exit_heads[str(self.config.early_exit_layers[-1])](
                        hidden_states
                    )
                    token_id = torch.argmax(exit_logits[:, -1, :], dim=-1).item()
                print(f"  ⚠ Forced generation (standalone mode)")
            
            # 检查是否结束
            if token_id == self.tokenizer.eos_token_id:
                break
            
            generated_tokens.append(token_id)
            
            # 更新输入
            input_ids = torch.tensor([[token_id]]).to(self.device)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=self.device)
            ], dim=1)
        
        total_time = time.time() - inference_start
        self.performance_stats['total_inference_time'].append(total_time)
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"\n✓ Generated {len(generated_tokens)} tokens in {total_time:.3f}s")
        print(f"  Throughput: {len(generated_tokens)/total_time:.2f} tokens/s")
        
        return generated_text
    
    async def generate_collaborative(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        云边协作推理模式（高精度模式）
        在边缘进行推理，必要时请求云端支持
        
        Args:
            prompt: 输入提示
            session_id: 会话ID
            max_new_tokens: 最大生成token数
        
        Returns:
            生成的文本
        """
        print("\n=== Cloud-Edge Collaborative Inference Mode ===")
        
        if session_id is None:
            session_id = f"session_{int(time.time()*1000)}"
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        generated_tokens = []
        past_key_values = None
        upload_task = None
        
        inference_start = time.time()
        
        for step in range(max_new_tokens):
            edge_start = time.time()
            
            # 边缘前向传播
            token_id, hidden_states, past_key_values, need_cloud = \
                self.edge_forward_with_early_exit(
                    input_ids,
                    attention_mask,
                    past_key_values
                )
            
            edge_time = time.time() - edge_start
            self.performance_stats['edge_inference_time'].append(edge_time)
            
            # 如果达到上传触发层，异步上传上下文
            if step == 0 or (step % 5 == 0 and self.config.async_upload):
                if upload_task is not None:
                    await upload_task  # 等待之前的上传完成
                
                upload_task = asyncio.create_task(
                    self.async_upload_context(
                        session_id,
                        hidden_states,
                        past_key_values
                    )
                )
            
            # 如果需要云端支持
            if need_cloud and self.config.mode == "collaborative":
                # 等待上传完成
                if upload_task is not None:
                    await upload_task
                
                # 请求云端推理
                all_input_ids = inputs.input_ids[0].tolist() + generated_tokens
                token_id = await self.request_cloud_inference(
                    session_id,
                    all_input_ids
                )
                
                if token_id is None:
                    # 云端请求失败，降级到边缘独立模式
                    print("  ⚠ Cloud request failed, fallback to edge")
                    with torch.no_grad():
                        exit_logits = self.exit_heads[str(self.config.early_exit_layers[-1])](
                            hidden_states
                        )
                        token_id = torch.argmax(exit_logits[:, -1, :], dim=-1).item()
            
            # 检查是否结束
            if token_id == self.tokenizer.eos_token_id:
                break
            
            generated_tokens.append(token_id)
            
            # 更新输入
            input_ids = torch.tensor([[token_id]]).to(self.device)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=self.device)
            ], dim=1)
        
        # 等待最后的上传完成
        if upload_task is not None:
            await upload_task
        
        total_time = time.time() - inference_start
        self.performance_stats['total_inference_time'].append(total_time)
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 打印统计信息
        stats = self.early_exit.get_statistics()
        print(f"\n✓ Generated {len(generated_tokens)} tokens in {total_time:.3f}s")
        print(f"  Throughput: {len(generated_tokens)/total_time:.2f} tokens/s")
        print(f"  Early exit rate: {stats['early_exit_rate']*100:.1f}%")
        print(f"  Cloud request rate: {stats['cloud_request_rate']*100:.1f}%")
        
        return generated_text
    
    def get_performance_summary(self) -> dict:
        """获取性能统计摘要"""
        summary = {}
        for key, values in self.performance_stats.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'total': np.sum(values)
                }
        
        summary['early_exit_stats'] = self.early_exit.get_statistics()
        return summary
