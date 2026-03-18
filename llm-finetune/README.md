# LLM Fine-tuning Project (大模型微调项目)

> **Parameter-efficient fine-tuning of large language models using LoRA / QLoRA**  
> A production-ready pipeline built on the Hugging Face ecosystem, demonstrating
> deep understanding of modern LLM training techniques.

---

## 📌 Project Highlights (面试亮点)

| 亮点 | 技术细节 |
|---|---|
| **参数高效微调 PEFT** | LoRA (r=16, alpha=32)，可训练参数仅占全量的 **0.1–1%**，显存节省 70%+ |
| **QLoRA 4-bit 量化** | 结合 bitsandbytes NF4 量化 + 双量化，7B 模型可在 **单张 24GB GPU** 上训练 |
| **指令微调数据** | 支持 Alpaca 与 ShareGPT 两种格式，含 **Loss Masking**（仅对 Response token 计算 loss）|
| **完整评估体系** | ROUGE-1/2/L + BLEU-1/2/3/4 自动评测，可扩展 perplexity |
| **工程化设计** | YAML 配置解耦、模块化架构、完整单元测试（pytest），支持 TensorBoard / W&B 监控 |
| **生产部署** | FastAPI REST 推理服务，支持流式 CLI 交互，LoRA adapter merge-and-unload |
| **分布式训练** | 兼容 `torchrun` 多卡训练 & DeepSpeed ZeRO-2/3 配置 |

---

## 🏗️ Architecture

```
llm-finetune/
├── configs/
│   ├── lora_config.yaml        # LoRA / QLoRA hyper-parameters
│   └── training_config.yaml    # Model, data, training, inference settings
├── data/
│   ├── sample_train.jsonl      # Sample Alpaca-format training data
│   └── sample_eval.jsonl       # Sample evaluation data
├── src/
│   ├── model/
│   │   └── model_utils.py      # Model loading, LoRA/QLoRA setup, inference merge
│   ├── data/
│   │   └── dataset.py          # InstructionDataset, DataCollator, prompt builders
│   ├── training/
│   │   └── trainer.py          # SFT trainer wrapper, callbacks, checkpoint saving
│   └── evaluation/
│       └── evaluator.py        # Perplexity, ROUGE, BLEU evaluation
├── scripts/
│   ├── train.py                # Training entry point
│   ├── inference.py            # CLI chat + FastAPI server
│   └── evaluate.py             # Batch evaluation script
└── tests/
    ├── test_dataset.py         # 17 unit tests for data pipeline
    ├── test_model_utils.py     # Unit tests for model utilities
    └── test_evaluator.py       # Unit tests for evaluation metrics
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your data

Data should be in **Alpaca format** (one JSON object per line):

```jsonl
{"instruction": "请用中文介绍LoRA微调技术", "input": "", "output": "LoRA（Low-Rank Adaptation）..."}
{"instruction": "翻译成英文", "input": "人工智能正在改变世界", "output": "Artificial intelligence is changing the world."}
```

Or **ShareGPT multi-turn format**:

```jsonl
{"conversations": [{"from": "human", "value": "你好"}, {"from": "assistant", "value": "你好！有什么可以帮助你的？"}]}
```

### 3. Configure

Edit `configs/training_config.yaml` to set your model and data paths:

```yaml
model:
  name_or_path: "Qwen/Qwen2-7B-Instruct"   # or meta-llama/Llama-2-7b-hf

data:
  train_file: "data/train.jsonl"
  eval_file: "data/eval.jsonl"
```

### 4. Train

**Single GPU (LoRA)**:
```bash
python scripts/train.py \
    --config configs/training_config.yaml \
    --lora_config configs/lora_config.yaml
```

**Single GPU (QLoRA – 4-bit, fits 24GB VRAM)**:
```bash
python scripts/train.py \
    --config configs/training_config.yaml \
    --lora_config configs/lora_config.yaml \
    --use_qlora
```

**Multi-GPU (4 GPUs with torchrun)**:
```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/training_config.yaml \
    --lora_config configs/lora_config.yaml
```

### 5. Inference

**Interactive CLI**:
```bash
python scripts/inference.py \
    --base_model Qwen/Qwen2-7B-Instruct \
    --adapter_path outputs/qwen2-7b-lora/final
```

**Single-shot**:
```bash
python scripts/inference.py \
    --base_model Qwen/Qwen2-7B-Instruct \
    --adapter_path outputs/qwen2-7b-lora/final \
    --instruction "请解释什么是注意力机制"
```

**FastAPI REST server**:
```bash
python scripts/inference.py \
    --base_model Qwen/Qwen2-7B-Instruct \
    --adapter_path outputs/qwen2-7b-lora/final \
    --serve --host 0.0.0.0 --port 8000
```

API example:
```bash
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"instruction": "用一句话描述Transformer架构", "max_new_tokens": 200}'
```

### 6. Evaluate

```bash
python scripts/evaluate.py \
    --base_model Qwen/Qwen2-7B-Instruct \
    --adapter_path outputs/qwen2-7b-lora/final \
    --eval_file data/eval.jsonl \
    --output_file results.json
```

---

## 🧠 Core Technical Concepts

### LoRA (Low-Rank Adaptation)

LoRA adds trainable **low-rank decomposition matrices** to the pre-trained weight matrices:

```
W' = W₀ + ΔW = W₀ + B·A
```

Where:
- `W₀ ∈ R^{d×k}` — frozen pre-trained weight
- `A ∈ R^{r×k}` — trainable, initialised with random Gaussian
- `B ∈ R^{d×r}` — trainable, initialised with zeros (so ΔW = 0 at the start)
- `r << min(d, k)` — rank, typically 8–64

At inference, the adapter is merged: `W' = W₀ + (α/r)·B·A`, adding **zero latency** overhead.

### QLoRA

QLoRA extends LoRA with:
1. **NF4 quantisation** of base model weights (4-bit NormalFloat)
2. **Double quantisation** of the quantisation constants (~0.4 bits/param savings)
3. **bfloat16 computation** — de-quantise to bf16 on the fly during forward/backward
4. **Paged optimiser** — handles gradient memory spikes via CUDA unified memory

Result: **7B model fine-tuning on a single 24GB GPU** with near full-precision quality.

### Loss Masking

During SFT, we only compute loss on **response tokens**, not prompt tokens:

```python
# labels = [-100, -100, ..., response_token_1, response_token_2, ...]
#            ←  prompt  →    ←         response          →
```

This prevents the model from learning to reproduce queries and focuses training signal entirely on generating high-quality responses.

---

## 📊 Supported Models

| Model | Size | Architecture | Language |
|---|---|---|---|
| `Qwen/Qwen2-7B-Instruct` | 7B | Transformer (GQA) | Chinese + English |
| `Qwen/Qwen2-72B-Instruct` | 72B | Transformer (GQA) | Chinese + English |
| `meta-llama/Llama-2-7b-hf` | 7B | Transformer (MHA) | English |
| `meta-llama/Llama-3-8B-Instruct` | 8B | Transformer (GQA) | English |
| `THUDM/chatglm3-6b` | 6B | GLM | Chinese + English |
| `mistralai/Mistral-7B-v0.1` | 7B | Transformer (GQA+SWA) | English |

---

## ⚙️ Configuration Reference

### `configs/lora_config.yaml`

```yaml
lora:
  r: 16                    # LoRA rank — higher = more capacity, more params
  lora_alpha: 32           # Scaling: effective lr ∝ alpha/r
  lora_dropout: 0.05       # Regularisation dropout
  target_modules: "all-linear"  # Apply LoRA to all linear layers
  bias: "none"             # Don't train bias terms
  task_type: "CAUSAL_LM"

qlora:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"          # NormalFloat4 recommended
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true
```

### `configs/training_config.yaml`

Key training arguments:

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4     # effective batch = 4×4 = 16
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"        # warmup + cosine decay
  gradient_checkpointing: true       # saves ~30% VRAM
  bf16: true
```

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Only data pipeline tests (no GPU needed)
pytest tests/test_dataset.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 📈 TensorBoard Monitoring

```bash
tensorboard --logdir outputs/qwen2-7b-lora/runs
```

---

## 🗂️ Data Format Details

### Alpaca format

```json
{
  "instruction": "任务描述（必填）",
  "input": "可选的补充输入或上下文",
  "output": "期望的模型输出（必填）"
}
```

### ShareGPT format

```json
{
  "conversations": [
    {"from": "human", "value": "用户消息"},
    {"from": "assistant", "value": "助手回复"},
    {"from": "human", "value": "追问"},
    {"from": "assistant", "value": "继续回复"}
  ]
}
```

---

## 🔧 Hardware Requirements

| Mode | GPU Memory | Recommended GPU |
|---|---|---|
| LoRA (bf16) | ~28 GB | A100-40G / 4090 |
| QLoRA (4-bit) | ~14 GB | 3090 / A10 / 4080 |
| QLoRA + gradient_checkpointing | ~10 GB | 3080 / A10 |
| Multi-GPU (4×) | 4 × 8 GB | 4 × 3080 |

---

## 📝 面试常见问题 Q&A

**Q: LoRA 为什么有效？**  
A: 预训练模型的权重更新矩阵具有低内在秩（intrinsic rank）的特点，因此可用低秩矩阵近似全量更新。LoRA 的核心创新在于将 ΔW 分解为 BA，B 初始化为零保证训练初期输出不变，A 随机初始化引入多样性。

**Q: QLoRA 和 LoRA 有什么区别？**  
A: QLoRA 在 LoRA 基础上引入了 4-bit NF4 量化，将基础模型的权重存储为 4-bit，前/反向传播时实时反量化到 bf16 计算。核心是 NormalFloat4 分布适配高斯分布的模型权重，比 INT4 精度更高。此外，双量化（double quantization）将量化常数本身也量化，每参数额外节省约 0.4 bits。

**Q: 为什么要做 Loss Masking？**  
A: 在指令微调中，Prompt 部分（instruction + input）是输入条件，模型不应"学习"复现它。若对 Prompt token 也计算 loss，会浪费训练信号（因为 Prompt 总是确定性的），并可能导致模型学到错误的分布。通过将 Prompt token 的 labels 设为 -100，PyTorch 的交叉熵损失会自动忽略这些位置。

**Q: 梯度检查点（gradient checkpointing）是什么原理？**  
A: 正向传播时不保存所有中间激活值，只保存部分"检查点"层的输出。反向传播时，从最近的检查点重新前向计算所需的激活值（以计算换内存）。复杂度从 O(n) 降至 O(√n)，通常可节省 30-40% 显存，代价是约 20-30% 的额外计算。

**Q: 如何选择 LoRA 的 target_modules？**  
A: 一般原则是对注意力层（q, k, v, o projections）和 MLP 层（gate, up, down projections）都应用 LoRA。使用 `"all-linear"` 是最简单的做法，会对所有线性层应用 LoRA。若显存受限，可只选 `["q_proj", "v_proj"]`，这是原论文中的推荐做法。

---

## 📚 References

- **LoRA**: [LORA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- **QLoRA**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
- **InstructGPT / RLHF**: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022)
- **Alpaca**: [Stanford Alpaca: An Instruction-following LLaMA Model](https://github.com/tatsu-lab/stanford_alpaca)
- **Hugging Face PEFT**: [Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
