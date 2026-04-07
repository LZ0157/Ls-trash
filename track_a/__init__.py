"""Track A: 硬冲突检测 — 基于句子分割 + NLI 的 CDConv baseline。

设计原则:
- CPU only (严格不使用 GPU)
- LLM 不参与任何步骤，全部用判别式 NLI 模型
- 模块化: data / segmenter / nli / pipeline / evaluate 可独立替换
"""
