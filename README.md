# TS-Diagnosis-Agent

独立的跨领域时间序列异常检测与知识辅助诊断实验仓库。

核心设计约束：异常解码器只参与源域训练期表示学习，**不是部署时异常分数来源**。部署评分通过正常解码器、目标域正常锚点和原始尺度证据计算；本地知识库与可选离线大模型消费结构化证据，不直接改写检测分数。

## 环境

Conda：
    conda env create -f environment.yml
    conda activate ts-diagnosis-agent

也可使用已有 Python：
    python -m pip install -r requirements.txt

environment.yml/requirements.txt 中的 PyTorch 安装应按本机 CUDA 版本调整；CPU 可直接运行全部演示流程。

## 合成数据端到端演示

PowerShell 中从仓库根目录运行：

    python scripts/make_synthetic_dataset.py --root data/synthetic
    python scripts/train_stage1.py --manifest data/synthetic/source_train.jsonl --val-manifest data/synthetic/source_val.jsonl --output artifacts/stage1 --epochs 20 --batch-size 64 --device cpu
    python scripts/adapt_stage2.py --pretrained artifacts/stage1/best.pt --normal-manifest data/synthetic/target_normal.jsonl --calibration-manifest data/synthetic/target_calibration.jsonl --unlabeled-manifest data/synthetic/target_unlabeled.jsonl --output artifacts/stage2 --epochs 10 --batch-size 64 --device cpu
    python scripts/evaluate.py --checkpoint artifacts/stage2/best.pt --normal-manifest data/synthetic/target_normal.jsonl --calibration-manifest data/synthetic/target_calibration.jsonl --test-manifest data/synthetic/target_test.jsonl --output artifacts/stage2/test_metrics.json --device cpu
    python scripts/diagnose_offline.py --checkpoint artifacts/stage2/best.pt --normal-manifest data/synthetic/target_normal.jsonl --calibration-manifest data/synthetic/target_calibration.jsonl --signal data/synthetic/signals/target_test_0001.npy --condition target --output artifacts/stage2/diagnosis_0001.json

如需调用本地 Ollama 模型（Ollama 服务需已在本机运行）：

    python scripts/diagnose_offline.py --checkpoint artifacts/stage2/best.pt --normal-manifest data/synthetic/target_normal.jsonl --calibration-manifest data/synthetic/target_calibration.jsonl --signal data/synthetic/signals/target_test_0001.npy --condition target --ollama-model YOUR_LOCAL_MODEL --output artifacts/stage2/diagnosis_llm.json

从本地 Neo4j 检索知识时，在当前 PowerShell 会话设置 NEO4J_URI、NEO4J_USER、NEO4J_PASSWORD，然后加 --neo4j。密码仅从环境变量读取，不要写入仓库或命令历史。

## 实验边界

- 阶段一使用多个源域的 source_train 和 source_val，训练共享/私有表征、正常解码器、训练期异常解码器、共享域对抗头和私有域分类头。
- 阶段二使用 target_normal、独立 target_calibration 和混合 target_unlabeled。适配只更新目标仿射适配器；伪正常候选按部署分数最低的固定分位筛选，脚本不读取无标签池的标签字段。
- target_calibration 只用于选择适配轮次与估计正常阈值，不参与梯度更新。
- target_test 只用于最终指标，不参与轮次、阈值或超参选择。评估器会阻止相同 sample_id/文件窗口跨目标集合重复。
- 目标域正常数据有限时，支持、校准数据都应按设备或记录独立拆分；不要把同一条记录切出的相邻窗口泄漏到两侧。

## 目录与数据

- tsdiag/：模型、清单读取、损失、评分、证据计算和知识检索。
- scripts/：数据生成、两阶段训练、评估和离线诊断。
- docs/architecture.md：结构与训练目标说明。
- docs/experiment_protocol.md：数据边界、基线和消融要求。
- data/README.md：JSONL 清单格式。
- knowledge/graph.json：未验证的示例知识图；正式研究必须替换为有出处并经专家核验的标准/手册知识。

输入窗口需预先保存为 .npy，形状 [channels, time] 或 [time]。同一批次窗口长度和通道数需一致。跨数据集实验前，需要统一窗口长度/采样率并记录来源、设备、工况和数据许可。

## 评分组成

默认初值：正常重建误差 0.45、多掩码误差波动 0.15、目标正常锚点距离 0.15、目标正常原始幅值尺度偏离 0.25。四项按目标正常统计量稳健归一化。系数只是可运行起点，须在验证/校准协议中预注册并冻结后再测试。测试集 F1 不用于找阈值；测试 F1 以目标正常校准集的预设误报分位阈值计算。

## 当前状态与限制

代码已通过合成数据端到端冒烟验证，目的是检查程序接口和阶段边界；生成的合成指标不代表真实数据性能。真实数据转成统一窗口清单、数据来源许可、经过验证的领域知识图谱、跨域多随机种子实验和正式 LLM 评测都还需要完成。演示知识图谱不是权威知识，不能直接用于真实维护决策。

