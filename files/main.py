"""
CE-CoLLM Main Entry Point
主程序入口，演示如何使用CE-CoLLM框架
"""

import asyncio
import argparse
from config import CECoLLMConfig
from edge_engine import EdgeInferenceEngine


async def demo_standalone_mode():
    """演示独立边缘推理模式"""
    print("\n" + "="*60)
    print("Demo: Standalone Edge Inference Mode (Low-Latency)")
    print("="*60)
    
    # 创建配置
    config = CECoLLMConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        mode="standalone",
        confidence_threshold=0.6,  # 降低阈值以便更多本地生成
        edge_num_layers=16,
        cloud_num_layers=16,
        early_exit_layers=[8, 16],
        max_new_tokens=50
    )
    
    # 创建边缘推理引擎
    engine = EdgeInferenceEngine(config)
    
    # 测试提示
    prompts = [
        "The capital of France is",
        "In machine learning, a neural network is",
        "The theory of relativity was proposed by"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Prompt: {prompt}")
        
        # 生成文本
        generated_text = await engine.generate_standalone(prompt)
        
        print(f"\nGenerated: {generated_text}")
    
    # 打印性能总结
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    summary = engine.get_performance_summary()
    
    if 'total_inference_time' in summary:
        print(f"Total inference time: {summary['total_inference_time']['total']:.3f}s")
        print(f"Average per request: {summary['total_inference_time']['mean']:.3f}s")
    
    if 'early_exit_stats' in summary:
        stats = summary['early_exit_stats']
        print(f"\nEarly Exit Statistics:")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Early exit rate: {stats['early_exit_rate']*100:.1f}%")
        print(f"  Exit layer distribution: {stats['exit_layer_counts']}")


async def demo_collaborative_mode():
    """演示云边协作推理模式"""
    print("\n" + "="*60)
    print("Demo: Cloud-Edge Collaborative Inference Mode (High-Accuracy)")
    print("="*60)
    print("\nNote: This demo requires a running cloud server.")
    print("Start the cloud server with: python cloud_server.py")
    print("="*60)
    
    # 创建配置
    config = CECoLLMConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        mode="collaborative",
        confidence_threshold=0.8,
        edge_num_layers=16,
        cloud_num_layers=16,
        early_exit_layers=[8, 16],
        max_new_tokens=50,
        cloud_server_url="http://localhost:8000",
        async_upload=True
    )
    
    # 创建边缘推理引擎
    engine = EdgeInferenceEngine(config)
    
    # 测试提示
    prompts = [
        "Explain the concept of cloud computing in simple terms:",
        "What are the benefits of edge computing?",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Prompt: {prompt}")
        
        try:
            # 生成文本
            generated_text = await engine.generate_collaborative(prompt)
            print(f"\nGenerated: {generated_text}")
        
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("Make sure the cloud server is running!")
    
    # 打印性能总结
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    summary = engine.get_performance_summary()
    
    if 'edge_inference_time' in summary and summary['edge_inference_time']:
        print(f"Edge inference time: {summary['edge_inference_time']['total']:.3f}s")
    
    if 'cloud_inference_time' in summary and summary['cloud_inference_time']:
        print(f"Cloud inference time: {summary['cloud_inference_time']['total']:.3f}s")
    
    if 'async_upload_time' in summary and summary['async_upload_time']:
        print(f"Async upload time: {summary['async_upload_time']['total']:.3f}s")
    
    if 'total_inference_time' in summary:
        print(f"Total inference time: {summary['total_inference_time']['total']:.3f}s")
    
    if 'early_exit_stats' in summary:
        stats = summary['early_exit_stats']
        print(f"\nEarly Exit Statistics:")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Early exit rate: {stats['early_exit_rate']*100:.1f}%")
        print(f"  Cloud request rate: {stats['cloud_request_rate']*100:.1f}%")


async def interactive_mode():
    """交互式模式"""
    print("\n" + "="*60)
    print("CE-CoLLM Interactive Mode")
    print("="*60)
    
    # 选择模式
    print("\nSelect inference mode:")
    print("1. Standalone (Low-Latency, Edge-only)")
    print("2. Collaborative (High-Accuracy, Cloud-Edge)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        mode = "standalone"
        print("\n✓ Using Standalone Mode")
    elif choice == "2":
        mode = "collaborative"
        print("\n✓ Using Collaborative Mode")
        print("Note: Make sure cloud server is running!")
    else:
        print("Invalid choice. Using standalone mode.")
        mode = "standalone"
    
    # 创建配置
    config = CECoLLMConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        mode=mode,
        confidence_threshold=0.8,
        max_new_tokens=100
    )
    
    # 创建引擎
    engine = EdgeInferenceEngine(config)
    
    print("\n" + "="*60)
    print("Enter your prompts (type 'quit' to exit)")
    print("="*60)
    
    while True:
        prompt = input("\nPrompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt:
            continue
        
        try:
            if mode == "standalone":
                generated = await engine.generate_standalone(prompt)
            else:
                generated = await engine.generate_collaborative(prompt)
            
            print(f"\nGenerated: {generated}")
        
        except Exception as e:
            print(f"\n✗ Error: {e}")
    
    # 打印最终统计
    print("\n" + "="*60)
    print("Session Summary")
    print("="*60)
    summary = engine.get_performance_summary()
    stats = summary.get('early_exit_stats', {})
    
    if stats.get('total_tokens', 0) > 0:
        print(f"Total tokens generated: {stats['total_tokens']}")
        print(f"Early exit rate: {stats.get('early_exit_rate', 0)*100:.1f}%")
        if mode == "collaborative":
            print(f"Cloud request rate: {stats.get('cloud_request_rate', 0)*100:.1f}%")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CE-CoLLM: Cloud-Edge Collaborative LLM")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['standalone', 'collaborative', 'interactive', 'demo'],
        default='demo',
        help='运行模式'
    )
    parser.add_argument(
        '--cloud-server',
        action='store_true',
        help='启动云端服务器'
    )
    
    args = parser.parse_args()
    
    if args.cloud_server:
        # 启动云端服务器
        from cloud_server import run_cloud_server
        from config import default_config
        run_cloud_server(default_config)
    
    else:
        # 运行客户端
        if args.mode == 'standalone':
            asyncio.run(demo_standalone_mode())
        
        elif args.mode == 'collaborative':
            asyncio.run(demo_collaborative_mode())
        
        elif args.mode == 'interactive':
            asyncio.run(interactive_mode())
        
        elif args.mode == 'demo':
            print("\n" + "="*60)
            print("CE-CoLLM Demo")
            print("="*60)
            print("\nRunning standalone mode demo...")
            asyncio.run(demo_standalone_mode())
            
            print("\n\n" + "="*60)
            print("For collaborative mode demo, start cloud server first:")
            print("  python main.py --cloud-server")
            print("\nThen in another terminal:")
            print("  python main.py --mode collaborative")
            print("="*60)


if __name__ == "__main__":
    main()
