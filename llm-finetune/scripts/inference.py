#!/usr/bin/env python3
"""
scripts/inference.py
====================
Interactive command-line inference using a fine-tuned LoRA model.

Usage
-----
Streaming CLI chat::

    python scripts/inference.py \\
        --base_model Qwen/Qwen2-7B-Instruct \\
        --adapter_path outputs/qwen2-7b-lora/final

Single-shot generation::

    python scripts/inference.py \\
        --base_model Qwen/Qwen2-7B-Instruct \\
        --adapter_path outputs/qwen2-7b-lora/final \\
        --instruction "请用一句话介绍LoRA微调技术"

Start a FastAPI REST endpoint::

    python scripts/inference.py \\
        --base_model Qwen/Qwen2-7B-Instruct \\
        --adapter_path outputs/qwen2-7b-lora/final \\
        --serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import build_alpaca_prompt
from src.evaluation.evaluator import generate_responses
from src.model.model_utils import load_lora_for_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned LoRA model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_model", type=str, required=True,
                        help="Hugging Face model ID or local path to the base model.")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to the saved LoRA adapter directory.")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Single instruction for non-interactive mode.")
    parser.add_argument("--input_text", type=str, default="",
                        help="Optional context/input for the instruction.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_sample", action="store_true",
                        help="Use greedy decoding instead of sampling.")
    # Server mode
    parser.add_argument("--serve", action="store_true",
                        help="Start a FastAPI REST API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(args):
    logger.info("Loading model and tokenizer…")
    model, tokenizer = load_lora_for_inference(
        args.base_model,
        args.adapter_path,
    )
    logger.info("Model loaded successfully.")
    return model, tokenizer


def run_single(args, model, tokenizer) -> str:
    prompt = build_alpaca_prompt(args.instruction, args.input_text)
    responses = generate_responses(
        model,
        tokenizer,
        [prompt],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.no_sample,
    )
    return responses[0]


def run_interactive(args, model, tokenizer) -> None:
    """Start an interactive REPL."""
    print("\n" + "=" * 60)
    print("  LLM Fine-tuning Project — Interactive Inference")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            instruction = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if instruction.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not instruction:
            continue

        input_text = input("Context (optional, press Enter to skip): ").strip()
        prompt = build_alpaca_prompt(instruction, input_text)

        print("\nAssistant: ", end="", flush=True)
        responses = generate_responses(
            model,
            tokenizer,
            [prompt],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.no_sample,
        )
        print(responses[0])
        print()


def run_server(args, model, tokenizer) -> None:
    """Start a FastAPI REST API server."""
    try:
        import uvicorn  # noqa: PLC0415
        from fastapi import FastAPI  # noqa: PLC0415
        from pydantic import BaseModel as PydanticBaseModel  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "FastAPI and uvicorn are required for server mode. "
            "Install with: pip install fastapi uvicorn"
        ) from exc

    app = FastAPI(
        title="LLM Fine-tuning Inference API",
        description="REST API for fine-tuned LLM inference",
        version="1.0.0",
    )

    class GenerateRequest(PydanticBaseModel):
        instruction: str
        input_text: str = ""
        max_new_tokens: int = 512
        temperature: float = 0.7
        top_p: float = 0.9
        top_k: int = 50
        repetition_penalty: float = 1.1
        do_sample: bool = True

    class GenerateResponse(PydanticBaseModel):
        response: str
        instruction: str

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest):
        prompt = build_alpaca_prompt(request.instruction, request.input_text)
        responses = generate_responses(
            model,
            tokenizer,
            [prompt],
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        return GenerateResponse(response=responses[0], instruction=request.instruction)

    logger.info("Starting FastAPI server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args)

    if args.serve:
        run_server(args, model, tokenizer)
    elif args.instruction:
        response = run_single(args, model, tokenizer)
        print(f"\n{'=' * 60}")
        print(f"Instruction: {args.instruction}")
        if args.input_text:
            print(f"Input:       {args.input_text}")
        print(f"Response:\n{response}")
        print("=" * 60)
    else:
        run_interactive(args, model, tokenizer)


if __name__ == "__main__":
    main()
