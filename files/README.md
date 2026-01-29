# CE-CoLLM: Cloud-Edge Collaborative Large Language Models

基于论文 "CE-CoLLM: Efficient and Adaptive Large Language Models Through Cloud-Edge Collaboration" 的完整实现。

## 📋 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [性能优化](#性能优化)
- [论文复现](#论文复现)

## 项目概述

CE-CoLLM是一个创新的云边协作框架，用于在边缘设备上高效部署大语言模型(LLM)。该框架通过以下关键技术解决了传统部署方式的痛点：

1. **延迟感知早退机制** - 在中间层终止推理，大幅减少计算量
2. **异步上下文上传** - 将数据传输与推理过程并行，掩盖通信延迟
3. **高效上下文管理** - 云端智能管理多个边缘客户端的状态
4. **双模式推理** - 支持独立边缘推理和云边协作推理

### 论文关键发现

根据论文实验：
- **通信开销降低99%+**: 从112MB降至<1MB每次响应
- **云端计算卸载84%+**: 大部分token在边缘生成
- **推理速度提升13.81%**: 相比纯云端部署
- **准确率保持一致**: 与完整云端LLM性能相当

## 核心特性

### 1. 延迟感知早退机制 (Latency-Aware Early Exit)

```python
# 在中间层设置早退点
early_exit_layers = [8, 16]  # 第8层和第16层

# 基于置信度动态决策
confidence_threshold = 0.8  # 高于0.8直接退出
```

**工作原理**:
- 在每个早退层计算token预测的置信度
- 如果置信度超过阈值，立即生成token，跳过后续层
- 论文发现：47.89%的token在中间层就能高置信度生成（Alpaca数据集）

### 2. 异步上下文上传 (Asynchronous Context Upload)

```python
# 在边缘推理的同时异步上传上下文到云端
async def async_upload_context(session_id, hidden_states):
    # 上传操作与推理并行
    await upload_to_cloud(session_id, hidden_states)
```

**优势**:
- 数据传输与边缘计算重叠
- 需要云支持时，上下文已经就绪
- FP16传输减少50%数据量

### 3. 云端上下文管理 (Cloud Context Management)

```python
class CloudContextManager:
    - 存储每个会话的隐藏状态
    - 维护KV缓存避免重复计算
    - 自动清理过期会话
```

### 4. 双模式推理

#### 独立边缘模式 (Standalone Mode)
- 完全在边缘设备运行
- 低延迟，不依赖网络
- 适合网络不稳定场景

#### 云边协作模式 (Collaborative Mode)  
- 边缘处理高置信度token
- 低置信度token请求云支持
- 高准确率，优化资源利用

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Edge Device                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Input Prompt → Tokenizer                              │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Edge Model Partition (Layers 0-15)                    │  │
│  │  ├─ Layer 0-7                                          │  │
│  │  ├─ Early Exit Point 1 (Layer 8) ──→ Confidence Check │  │
│  │  ├─ Layer 9-15                                         │  │
│  │  └─ Early Exit Point 2 (Layer 16) ──→ Confidence Check│  │
│  └───────────────────────────────────────────────────────┘  │
│           ↓ (if conf ≥ 0.8)          ↓ (if conf < 0.8)     │
│    Generate Token Locally      Request Cloud Support        │
│                                       ↓                      │
└───────────────────────────────────────┼──────────────────────┘
                                        │ Async Upload Context
                                        ↓
┌─────────────────────────────────────────────────────────────┐
│                      Cloud Server                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Context Manager                                       │  │
│  │  ├─ Session Contexts                                   │  │
│  │  ├─ KV Cache Storage                                   │  │
│  │  └─ Timeout Management                                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Cloud Model Partition (Layers 16-31)                  │  │
│  │  └─ Continue from Hidden States                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│                   Generate Next Token                        │
│                           ↓                                  │
└───────────────────────────┼──────────────────────────────────┘
                            │ Return Single Token
                            ↓
                      Edge Device
```

## 安装指南

### 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐用于GPU加速)
- 至少16GB RAM (边缘设备)
- 至少32GB VRAM (云端服务器，用于完整LLM)

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/yourusername/CE-CoLLM.git
cd CE-CoLLM
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **下载预训练模型**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

## 快速开始

### 方式1: 运行完整Demo

```bash
python main.py --mode demo
```

这将运行独立模式的演示，展示基本功能。

### 方式2: 云边协作模式

**终端1 - 启动云端服务器:**
```bash
python main.py --cloud-server
```

**终端2 - 运行边缘客户端:**
```bash
python main.py --mode collaborative
```

### 方式3: 交互式模式

```bash
python main.py --mode interactive
```

然后根据提示选择推理模式并输入您的问题。

## 详细使用

### 配置系统

编辑 `config.py` 来自定义系统参数：

```python
config = CECoLLMConfig(
    # 模型设置
    model_name="meta-llama/Llama-2-7b-hf",
    num_layers=32,
    
    # 云边分割
    edge_num_layers=16,      # 边缘设备上的层数
    cloud_num_layers=16,     # 云端的层数
    
    # 早退设置
    early_exit_layers=[8, 16],  # 早退点位置
    confidence_threshold=0.8,     # 置信度阈值
    
    # 通信优化
    use_fp16_transfer=True,   # 使用FP16减少传输量
    async_upload=True,        # 启用异步上传
    
    # 推理参数
    max_new_tokens=100,
    temperature=1.0,
    top_p=0.9,
    
    # 运行模式
    mode="collaborative"  # "standalone" 或 "collaborative"
)
```

### 使用独立模式

```python
import asyncio
from config import CECoLLMConfig
from edge_engine import EdgeInferenceEngine

async def main():
    config = CECoLLMConfig(mode="standalone")
    engine = EdgeInferenceEngine(config)
    
    prompt = "Explain quantum computing:"
    result = await engine.generate_standalone(prompt)
    print(result)

asyncio.run(main())
```

### 使用协作模式

```python
async def main():
    config = CECoLLMConfig(mode="collaborative")
    engine = EdgeInferenceEngine(config)
    
    prompt = "What is machine learning?"
    result = await engine.generate_collaborative(prompt)
    print(result)

asyncio.run(main())
```

### 训练早退头

如果您想训练自己的早退头：

```python
from early_exit import EarlyExitLLM, train_early_exit_heads
from transformers import AutoModelForCausalLM
import torch

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 创建带早退的模型
model = EarlyExitLLM(
    base_model=base_model,
    early_exit_layers=[8, 16],
    hidden_size=4096,
    vocab_size=32000
)

# 准备数据加载器
# dataloader = ...

# 训练
optimizer = torch.optim.Adam(model.exit_heads.parameters(), lr=1e-4)
train_early_exit_heads(model, dataloader, optimizer, num_epochs=3)
```

## 性能优化

### 1. 调整早退阈值

```python
# 更激进的早退（更快，可能略微降低准确率）
confidence_threshold = 0.7

# 更保守的早退（更准确，但可能需要更多云请求）
confidence_threshold = 0.9
```

### 2. 优化云边分割

```python
# 边缘设备性能强：更多层在边缘
edge_num_layers = 20
cloud_num_layers = 12

# 边缘设备性能弱：更多层在云端
edge_num_layers = 12
cloud_num_layers = 20
```

### 3. 使用量化加速

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

### 4. 批处理推理

```python
config = CECoLLMConfig(batch_size=4)  # 同时处理4个请求
```

## 论文复现

### 复现论文中的实验

#### 实验1: 通信开销比较

```bash
python experiments/communication_overhead.py
```

预期输出：
- Naive Cloud-Edge: ~112MB per response
- CE-CoLLM: <1MB per response
- 减少 >99%

#### 实验2: 推理时间比较

```bash
python experiments/inference_time.py \
    --dataset alpaca \
    --samples 100
```

预期输出（Alpaca数据集）：
- Cloud LLM: ~370s
- CE-CoLLM: ~319s
- 提升: 13.81%

#### 实验3: 准确率评估

```bash
python experiments/accuracy_eval.py \
    --task boolq \
    --mode collaborative
```

预期输出：
- CE-CoLLM (collaborative): 0.658 (EM)
- Cloud LLM: 0.646 (EM)
- 准确率保持一致

### 数据集

论文使用的数据集：
- **Alpaca**: 指令遵循任务
- **XSum**: 文本摘要
- **BoolQ**: 问答（是非题）
- **QuAC**: 对话问答
- **IMDB**: 情感分析

下载数据集：
```bash
python scripts/download_datasets.py
```

## 项目结构

```
CE-CoLLM/
├── config.py              # 配置文件
├── early_exit.py          # 早退机制实现
├── edge_engine.py         # 边缘推理引擎
├── cloud_server.py        # 云端服务器
├── main.py                # 主程序入口
├── requirements.txt       # 依赖列表
├── README.md             # 本文档
├── experiments/          # 论文实验脚本
│   ├── communication_overhead.py
│   ├── inference_time.py
│   └── accuracy_eval.py
├── scripts/              # 辅助脚本
│   └── download_datasets.py
└── tests/                # 单元测试
    ├── test_early_exit.py
    ├── test_edge_engine.py
    └── test_cloud_server.py
```

## 常见问题

### Q1: 如何处理网络不稳定的情况？

使用独立模式：
```python
config = CECoLLMConfig(mode="standalone")
```

### Q2: 云端请求失败怎么办？

系统会自动降级到边缘独立推理：
```python
# 在edge_engine.py中已实现自动fallback
if token_id is None:  # 云端失败
    # 使用边缘模型强制生成
    token_id = edge_generate_fallback()
```

### Q3: 如何减少内存使用？

1. 使用量化模型
2. 减少batch_size
3. 限制KV缓存大小

### Q4: 准确率下降怎么办？

1. 提高confidence_threshold
2. 增加early_exit_layers
3. 训练更好的早退头

## 性能基准

基于LLaMA-2-7B模型：

| 指标 | Cloud LLM | Naive Cloud-Edge | CE-CoLLM |
|------|-----------|------------------|----------|
| 推理时间 (Alpaca) | 370s | 3372s | **319s** |
| 通信数据量 | 367KB | 112MB | **957KB** |
| 云端请求率 | 100% | 100% | **49.58%** |
| 准确率 (BoolQ) | 0.646 | - | 0.658 |

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 引用

如果您使用了这个实现，请引用原论文：

```bibtex
@article{jin2024cecollm,
  title={CE-CoLLM: Efficient and Adaptive Large Language Models Through Cloud-Edge Collaboration},
  author={Jin, Hongpeng and Wu, Yanzhao},
  journal={arXiv preprint arXiv:2411.02829},
  year={2024}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢

- 原论文作者: Hongpeng Jin, Yanzhao Wu (Florida International University)
- Transformers库: Hugging Face
- LLaMA模型: Meta AI

## 联系方式

如有问题或建议，请：
- 提交 Issue
- 发送邮件至: your.email@example.com
- 加入讨论群: [链接]

---

**⭐ 如果这个项目对您有帮助，请给我们一个star！**
