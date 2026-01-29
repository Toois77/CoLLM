"""
Communication Overhead Experiment
复现论文中关于通信开销的实验结果
"""

import torch
import numpy as np
import time
from typing import List, Tuple
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import json

from config import CECoLLMConfig


class CommunicationOverheadAnalyzer:
    """分析不同部署策略的通信开销"""
    
    def __init__(self, config: CECoLLMConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
    def estimate_data_size(self, tensor: torch.Tensor, use_fp16: bool = False) -> float:
        """
        估算张量的传输大小（KB）
        
        Args:
            tensor: PyTorch张量
            use_fp16: 是否使用FP16
        
        Returns:
            size_kb: 大小（KB）
        """
        num_elements = tensor.numel()
        bytes_per_element = 2 if use_fp16 else 4  # FP16: 2字节, FP32: 4字节
        size_bytes = num_elements * bytes_per_element
        size_kb = size_bytes / 1024
        return size_kb
    
    def simulate_naive_cloud_edge(
        self,
        prompt: str,
        num_tokens: int = 100
    ) -> Tuple[float, List[float]]:
        """
        模拟Naive Cloud-Edge部署的通信开销
        每个token都需要传输完整的隐藏状态
        
        Returns:
            total_kb: 总传输量（KB）
            per_token_kb: 每个token的传输量列表
        """
        # 编码prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        seq_len = inputs.input_ids.shape[1]
        
        per_token_sizes = []
        
        for i in range(num_tokens):
            # 隐藏状态: [batch_size, seq_len, hidden_size]
            current_seq_len = seq_len + i
            hidden_states = torch.randn(1, current_seq_len, self.config.hidden_size)
            
            # 计算传输大小
            size_kb = self.estimate_data_size(hidden_states, use_fp16=False)
            per_token_sizes.append(size_kb)
        
        total_kb = sum(per_token_sizes)
        return total_kb, per_token_sizes
    
    def simulate_ce_collm(
        self,
        prompt: str,
        num_tokens: int = 100,
        cloud_request_rate: float = 0.5  # 论文中约50%请求云端
    ) -> Tuple[float, List[float], dict]:
        """
        模拟CE-CoLLM的通信开销
        
        Returns:
            total_kb: 总传输量（KB）
            per_request_kb: 每次云请求的传输量
            stats: 统计信息
        """
        # 编码prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        seq_len = inputs.input_ids.shape[1]
        
        # 计算需要云支持的token数量
        num_cloud_requests = int(num_tokens * cloud_request_rate)
        
        # 异步上传的数据（只传输边缘分区的隐藏状态）
        upload_sizes = []
        request_sizes = []
        
        upload_interval = 5  # 每5个token上传一次
        num_uploads = (num_tokens + upload_interval - 1) // upload_interval
        
        for i in range(num_uploads):
            # 上传边缘分区的隐藏状态
            # [batch_size, current_seq_len, hidden_size]
            current_seq_len = min(seq_len + (i+1) * upload_interval, seq_len + num_tokens)
            hidden_states = torch.randn(1, current_seq_len, self.config.hidden_size)
            
            # 使用FP16减少传输量
            size_kb = self.estimate_data_size(hidden_states, use_fp16=True)
            upload_sizes.append(size_kb)
        
        # 云请求只传输token ID和少量元数据
        for i in range(num_cloud_requests):
            # 只传输input_ids列表（很小）
            input_ids_size = (seq_len + i) * 4 / 1024  # 每个token ID 4字节
            request_sizes.append(input_ids_size)
        
        total_kb = sum(upload_sizes) + sum(request_sizes)
        
        stats = {
            'num_uploads': num_uploads,
            'num_cloud_requests': num_cloud_requests,
            'avg_upload_size': np.mean(upload_sizes) if upload_sizes else 0,
            'total_upload_size': sum(upload_sizes),
            'total_request_size': sum(request_sizes),
            'cloud_request_rate': cloud_request_rate
        }
        
        return total_kb, request_sizes, stats
    
    def run_comparison(
        self,
        prompts: List[str],
        num_tokens: int = 100
    ) -> dict:
        """
        运行完整的通信开销比较实验
        
        Args:
            prompts: 测试提示列表
            num_tokens: 每个提示生成的token数
        
        Returns:
            results: 实验结果
        """
        print("\n" + "="*70)
        print("Communication Overhead Comparison Experiment")
        print("="*70)
        
        naive_results = []
        ce_collm_results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- Prompt {i}/{len(prompts)} ---")
            print(f"Prompt: {prompt[:50]}...")
            
            # Naive Cloud-Edge
            naive_total, naive_per_token = self.simulate_naive_cloud_edge(
                prompt, num_tokens
            )
            naive_avg = naive_total / num_tokens
            
            print(f"\nNaive Cloud-Edge:")
            print(f"  Total data: {naive_total:.2f} KB ({naive_total/1024:.2f} MB)")
            print(f"  Avg per token: {naive_avg:.2f} KB")
            
            naive_results.append({
                'total_kb': naive_total,
                'avg_per_token': naive_avg
            })
            
            # CE-CoLLM (使用论文中的云请求率)
            ce_total, ce_per_request, ce_stats = self.simulate_ce_collm(
                prompt, num_tokens, cloud_request_rate=0.5
            )
            ce_avg = ce_total / num_tokens if num_tokens > 0 else 0
            
            print(f"\nCE-CoLLM:")
            print(f"  Total data: {ce_total:.2f} KB")
            print(f"  Avg per token: {ce_avg:.2f} KB")
            print(f"  Cloud requests: {ce_stats['num_cloud_requests']}/{num_tokens} " +
                  f"({ce_stats['cloud_request_rate']*100:.1f}%)")
            print(f"  Data reduction: {(1 - ce_total/naive_total)*100:.2f}%")
            
            ce_collm_results.append({
                'total_kb': ce_total,
                'avg_per_token': ce_avg,
                'stats': ce_stats,
                'reduction_rate': (1 - ce_total/naive_total)
            })
        
        # 汇总结果
        results = {
            'naive_cloud_edge': {
                'avg_total_kb': np.mean([r['total_kb'] for r in naive_results]),
                'avg_per_token_kb': np.mean([r['avg_per_token'] for r in naive_results])
            },
            'ce_collm': {
                'avg_total_kb': np.mean([r['total_kb'] for r in ce_collm_results]),
                'avg_per_token_kb': np.mean([r['avg_per_token'] for r in ce_collm_results]),
                'avg_cloud_request_rate': np.mean([r['stats']['cloud_request_rate'] 
                                                    for r in ce_collm_results]),
                'avg_reduction_rate': np.mean([r['reduction_rate'] for r in ce_collm_results])
            }
        }
        
        # 打印汇总
        print("\n" + "="*70)
        print("Summary Statistics")
        print("="*70)
        print(f"\nNaive Cloud-Edge:")
        print(f"  Average total: {results['naive_cloud_edge']['avg_total_kb']:.2f} KB " +
              f"({results['naive_cloud_edge']['avg_total_kb']/1024:.2f} MB)")
        print(f"  Average per response: {results['naive_cloud_edge']['avg_per_token_kb']:.2f} KB")
        
        print(f"\nCE-CoLLM:")
        print(f"  Average total: {results['ce_collm']['avg_total_kb']:.2f} KB")
        print(f"  Average per response: {results['ce_collm']['avg_per_token_kb']:.2f} KB")
        print(f"  Cloud request rate: {results['ce_collm']['avg_cloud_request_rate']*100:.2f}%")
        print(f"  Data reduction: {results['ce_collm']['avg_reduction_rate']*100:.2f}%")
        
        return results
    
    def visualize_results(self, results: dict, save_path: str = "communication_overhead.png"):
        """可视化实验结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 图1: 总数据传输量对比
        methods = ['Naive\nCloud-Edge', 'CE-CoLLM']
        total_data = [
            results['naive_cloud_edge']['avg_total_kb'] / 1024,  # 转换为MB
            results['ce_collm']['avg_total_kb'] / 1024
        ]
        
        bars1 = ax1.bar(methods, total_data, color=['#e74c3c', '#2ecc71'])
        ax1.set_ylabel('Average Data Transfer (MB)', fontsize=12)
        ax1.set_title('Total Communication Overhead', fontsize=14, fontweight='bold')
        ax1.set_yscale('log')
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f} MB',
                    ha='center', va='bottom', fontsize=10)
        
        # 图2: 云请求率
        cloud_request_rates = [
            100,  # Naive方法100%请求云端
            results['ce_collm']['avg_cloud_request_rate'] * 100
        ]
        
        bars2 = ax2.bar(methods, cloud_request_rates, color=['#e74c3c', '#2ecc71'])
        ax2.set_ylabel('Cloud Request Rate (%)', fontsize=12)
        ax2.set_title('Cloud Request Frequency', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 110])
        
        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved to {save_path}")


def main():
    """运行通信开销实验"""
    
    # 配置
    config = CECoLLMConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        hidden_size=4096,
        edge_num_layers=16,
        cloud_num_layers=16
    )
    
    # 创建分析器
    analyzer = CommunicationOverheadAnalyzer(config)
    
    # 测试提示（模拟Alpaca风格的短提示）
    short_prompts = [
        "What is the capital of France?",
        "Explain what is photosynthesis.",
        "Write a haiku about summer.",
        "List three benefits of exercise.",
        "Define machine learning in simple terms."
    ]
    
    # 测试提示（模拟XSum风格的长提示）
    long_prompts = [
        "Summarize the following article: " + "Climate change is a pressing global issue. " * 50,
        "Analyze the key points of this document: " + "Economic policy affects many aspects. " * 50,
    ]
    
    print("\n" + "="*70)
    print("Testing with SHORT prompts (Alpaca-style)")
    print("="*70)
    results_short = analyzer.run_comparison(short_prompts, num_tokens=100)
    
    print("\n" + "="*70)
    print("Testing with LONG prompts (XSum-style)")
    print("="*70)
    results_long = analyzer.run_comparison(long_prompts, num_tokens=100)
    
    # 可视化结果
    analyzer.visualize_results(results_short, "communication_overhead_short.png")
    analyzer.visualize_results(results_long, "communication_overhead_long.png")
    
    # 保存结果到JSON
    all_results = {
        'short_prompts': results_short,
        'long_prompts': results_long,
        'config': {
            'hidden_size': config.hidden_size,
            'edge_num_layers': config.edge_num_layers,
            'cloud_num_layers': config.cloud_num_layers
        }
    }
    
    with open('communication_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n✓ Results saved to communication_results.json")


if __name__ == "__main__":
    main()
