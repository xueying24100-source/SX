# SX Projects

This repository contains multiple projects including an LLM Fine-tuning project and a To-Do List application.

## Projects

### 🤖 [LLM Fine-tuning Project](./llm-finetune/README.md)

A production-ready pipeline for parameter-efficient fine-tuning of large language models (7B–72B) using **LoRA** and **QLoRA** with the Hugging Face ecosystem.

**Key features:**
- LoRA / QLoRA fine-tuning (reduces trainable params by 99%+)
- Support for Qwen2, LLaMA-2/3, ChatGLM, Mistral models
- Alpaca & ShareGPT instruction-following data formats
- Loss masking, gradient checkpointing, mixed-precision training
- ROUGE + BLEU automatic evaluation
- FastAPI REST inference endpoint
- Multi-GPU training via `torchrun` + DeepSpeed ZeRO support

See [`llm-finetune/README.md`](./llm-finetune/README.md) for detailed documentation.