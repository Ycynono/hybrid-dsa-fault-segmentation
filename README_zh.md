# Hybrid DSA 三维地震断层分割复现包

这是论文《三维地震断层分割中的标注域与跨工区稳健性：一种紧凑型 Hybrid DSA 网络》的官方复现包。

仓库包含完整模型源码、解析型合成数据生成器、数据预处理、训练与评估脚本、冻结协议、最终模型权重、机器可读结果及绘图程序。F3、Thebe、CRACKS、Smeaheia、FORCE 和 Delft 等第三方地震数据不在仓库中重新分发。

## 快速验证

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_release.py
python synthetic_fault_generator.py --output processed_data/smoke --n-samples 3 --train 1 --val 1 --shape 32,32,32 --qc-count 0
python -m fault_experiments.smoke_test --data-root processed_data/smoke --model unet3d --base-channels 8
python -m fault_experiments.smoke_test --data-root processed_data/smoke --model dsa_hybrid --base-channels 8
```

完整合成训练、真实数据准备、冻结阈值和结果复现命令见英文 [`README.md`](README.md)。数据下载地址、许可和本地目录规范见 [`docs/DATA.md`](docs/DATA.md)。

## 固定协议

- 合成数据随机种子：`20261101`；300/50/50 划分；体尺寸 `128 x 128 x 128`。
- 所有模型依次经过合成预训练、F3 稀疏适配和 Thebe 适配。
- 检查点与阈值只由 Thebe val1-val2 选择。
- U-Net、Hybrid DSA、SwinUNETR 阈值分别为 `0.50`、`0.15`、`0.40`。
- Thebe test2-test7 为冻结测试；CRACKS 由 20 个审计剖面与 20 个一次性密封保留剖面组成。
- Smeaheia GN1101 为独立稀疏三维专家验证；确定性 ROI 含 36 个断层对象和 297 个专家剖面，专家有效走廊外不视为负样本。
- FORCE 与 Delft 没有独立人工标签，只用于跨工区校准失效分析，不能报告为准确率。

Smeaheia 下载、配准审计、ROI 准备、冻结推理、相干对照、评价和三维图件可按以下顺序复现：

```powershell
python scripts/download_smeaheia.py fault_sticks reports seismic_3d --country "<国家>" --affiliation "<单位>"
python -m fault_experiments.audit_smeaheia_dataset --require-segy
python -m fault_experiments.prepare_smeaheia_benchmark
python -m fault_experiments.run_smeaheia_frozen_external
python -m fault_experiments.run_smeaheia_coherence
python -m fault_experiments.evaluate_smeaheia_benchmark
python -m fault_experiments.evaluate_smeaheia_fault_objects
python -m fault_experiments.visualize_smeaheia_comparison
```

代码采用 MIT License；第三方数据仍受原始提供方许可约束。
