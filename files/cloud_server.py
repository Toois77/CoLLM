"""
Cloud Server for CE-CoLLM
云端服务器，负责上下文管理和继续推理
"""

import torch
import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass
import time
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

from config import CECoLLMConfig


@dataclass
class CloudContext:
    """云端存储的上下文"""
    session_id: str
    hidden_states: torch.Tensor
    past_key_values: Optional[tuple]
    last_update_time: float
    

class ContextRequest(BaseModel):
    """上下文上传请求"""
    session_id: str
    hidden_states: bytes
    hidden_states_shape: list
    dtype: str
    has_kv_cache: bool = False


class InferenceRequest(BaseModel):
    """推理请求"""
    session_id: str
    input_ids: list
    temperature: float = 1.0
    top_p: float = 0.9


class CloudContextManager:
    """
    云端上下文管理器
    管理多个边缘客户端的上下文数据
    """
    
    def __init__(self, config: CECoLLMConfig):
        self.config = config
        self.contexts: Dict[str, CloudContext] = {}
        self.session_timeout = config.session_timeout
        
        # 性能统计
        self.stats = defaultdict(int)
    
    def store_context(
        self,
        session_id: str,
        hidden_states: torch.Tensor,
        past_key_values: Optional[tuple] = None
    ):
        """
        存储边缘上传的上下文
        
        Args:
            session_id: 会话ID
            hidden_states: 隐藏状态
            past_key_values: KV缓存
        """
        self.contexts[session_id] = CloudContext(
            session_id=session_id,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            last_update_time=time.time()
        )
        self.stats['contexts_stored'] += 1
        print(f"✓ Context stored for session: {session_id}")
    
    def get_context(self, session_id: str) -> Optional[CloudContext]:
        """
        获取指定会话的上下文
        
        Args:
            session_id: 会话ID
        
        Returns:
            上下文数据或None
        """
        if session_id not in self.contexts:
            return None
        
        context = self.contexts[session_id]
        
        # 检查是否超时
        if time.time() - context.last_update_time > self.session_timeout:
            self.delete_context(session_id)
            return None
        
        self.stats['contexts_retrieved'] += 1
        return context
    
    def delete_context(self, session_id: str):
        """删除上下文"""
        if session_id in self.contexts:
            del self.contexts[session_id]
            self.stats['contexts_deleted'] += 1
            print(f"✓ Context deleted for session: {session_id}")
    
    def cleanup_expired_contexts(self):
        """清理过期的上下文"""
        current_time = time.time()
        expired_sessions = [
            sid for sid, ctx in self.contexts.items()
            if current_time - ctx.last_update_time > self.session_timeout
        ]
        
        for session_id in expired_sessions:
            self.delete_context(session_id)
        
        if expired_sessions:
            print(f"✓ Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            'active_sessions': len(self.contexts),
            **dict(self.stats)
        }


class CloudInferenceServer:
    """
    云端推理服务器
    处理边缘设备的推理请求
    """
    
    def __init__(self, config: CECoLLMConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 加载云端模型（后N层）
        print(f"Loading cloud model ({config.cloud_num_layers} layers)...")
        self.cloud_model = self._load_cloud_model()
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 上下文管理器
        self.context_manager = CloudContextManager(config)
        
        # 性能统计
        self.inference_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'inference_times': []
        }
    
    def _load_cloud_model(self) -> torch.nn.Module:
        """
        加载云端模型分区（后N层）
        实际实现中需要只加载后N层
        """
        # 加载完整模型
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        
        # 在实际实现中，这里应该只保留后cloud_num_layers层
        # 简化版本：使用完整模型
        
        model.eval()
        return model
    
    def continue_inference(
        self,
        session_id: str,
        input_ids: list,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> Optional[int]:
        """
        继续推理，生成下一个token
        
        Args:
            session_id: 会话ID
            input_ids: 完整的输入token序列
            temperature: 温度参数
            top_p: nucleus sampling参数
        
        Returns:
            next_token_id: 生成的下一个token
        """
        inference_start = time.time()
        self.inference_stats['total_requests'] += 1
        
        try:
            # 获取上下文
            context = self.context_manager.get_context(session_id)
            if context is None:
                print(f"✗ Context not found for session: {session_id}")
                self.inference_stats['failed_requests'] += 1
                return None
            
            # 准备输入
            input_tensor = torch.tensor([input_ids]).to(self.device)
            
            # 使用云端模型继续推理
            with torch.no_grad():
                # 在实际实现中，应该从context.hidden_states继续推理
                # 这里简化为使用完整模型
                outputs = self.cloud_model(
                    input_ids=input_tensor,
                    past_key_values=context.past_key_values,
                    use_cache=True
                )
                
                logits = outputs.logits[:, -1, :]  # 最后一个位置的logits
                
                # 应用temperature
                logits = logits / temperature
                
                # 应用top-p采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    
                    # 移除累积概率超过top_p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # 采样
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # 更新KV缓存
                context.past_key_values = outputs.past_key_values
                context.last_update_time = time.time()
            
            inference_time = time.time() - inference_start
            self.inference_stats['inference_times'].append(inference_time)
            self.inference_stats['successful_requests'] += 1
            
            print(f"✓ Cloud inference completed in {inference_time:.3f}s, token: {next_token}")
            return next_token
        
        except Exception as e:
            print(f"✗ Cloud inference error: {e}")
            self.inference_stats['failed_requests'] += 1
            return None
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        stats = self.inference_stats.copy()
        if stats['inference_times']:
            stats['avg_inference_time'] = np.mean(stats['inference_times'])
            stats['min_inference_time'] = np.min(stats['inference_times'])
            stats['max_inference_time'] = np.max(stats['inference_times'])
        
        stats['context_manager'] = self.context_manager.get_statistics()
        return stats


# FastAPI应用
def create_cloud_server_app(config: CECoLLMConfig) -> FastAPI:
    """创建云端服务器FastAPI应用"""
    
    app = FastAPI(title="CE-CoLLM Cloud Server")
    server = CloudInferenceServer(config)
    
    @app.post("/context/upload")
    async def upload_context(request: ContextRequest):
        """上传上下文数据"""
        try:
            # 反序列化隐藏状态
            dtype = np.float16 if request.dtype == 'float16' else np.float32
            hidden_states_array = np.frombuffer(
                request.hidden_states,
                dtype=dtype
            ).reshape(request.hidden_states_shape)
            
            hidden_states = torch.from_numpy(hidden_states_array).to(server.device)
            
            # 存储上下文
            server.context_manager.store_context(
                request.session_id,
                hidden_states,
                past_key_values=None  # 简化处理
            )
            
            return {"status": "success", "session_id": request.session_id}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/inference/continue")
    async def continue_inference(request: InferenceRequest):
        """继续推理"""
        try:
            next_token = server.continue_inference(
                request.session_id,
                request.input_ids,
                request.temperature,
                request.top_p
            )
            
            if next_token is None:
                raise HTTPException(status_code=404, detail="Context not found or inference failed")
            
            return {"next_token": next_token}
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/statistics")
    async def get_statistics():
        """获取服务器统计信息"""
        return server.get_statistics()
    
    @app.post("/cleanup")
    async def cleanup_contexts():
        """清理过期上下文"""
        server.context_manager.cleanup_expired_contexts()
        return {"status": "success"}
    
    return app


def run_cloud_server(config: CECoLLMConfig, host: str = "0.0.0.0", port: int = 8000):
    """运行云端服务器"""
    app = create_cloud_server_app(config)
    print(f"\n=== Starting CE-CoLLM Cloud Server ===")
    print(f"Server URL: http://{host}:{port}")
    print(f"Model: {config.model_name}")
    print(f"Cloud layers: {config.cloud_num_layers}")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    from config import default_config
    run_cloud_server(default_config)
