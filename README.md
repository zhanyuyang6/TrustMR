# TrustMR

**TrustMR：多模态数字证据的联合固证与可验证检索**

TrustMR 的实验实现，包括多模态向量检索、联合承诺、查询结果验证、转录绑定随机审计，以及 V-GANN、VSQ 对比实验。

## 实现与配置

论文实验采用 `src/protocol_compact.py` 和 `src/compact_wire.py`。该实现将记录头与完整向量分离，验证方核验探查范围内的记录，并对粗筛选中的完整向量重新计算排序。代码中的 **B2 compact** 是这一实现的内部版本名。

| 参数 | 主对比配置 |
| --- | --- |
| 数据 | MSCOCO 2017，118,287 张训练集图像 |
| 编码 | ImageBind，1,024 维向量 |
| 二进制签名 | 256 位，未加权 Hamming 距离 |
| 桶前缀长度 `p` | 8 |
| 翻转位数 `mmax` | 8 |
| 粗筛数量 `Kcoarse` | 1,000 |
| 返回数量 `K` | 20 |
| 主对比查询范围 | 无时间、来源过滤，目标模态为图像 |

`p=8, mmax=8` 时，主检索覆盖全部 256 个桶；有限探查和随机审计使用独立配置。`src/protocol_v6.py` 提供共用定义与全向量参考实现，主对比不使用其全向量传输格式。

## 安装与测试

实验环境为 Ubuntu 22.04、Python 3.10.20、NumPy 1.26.4。论文所用服务器为浪潮 NF5280M6，配置两颗 Xeon Gold 5318Y、512 GB 内存和两张 NVIDIA A800 80 GB。协议单元测试可在不安装 GPU 编码环境的情况下运行。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/protocol_compact.py` | 记录头、选择性向量响应、核验与审计 |
| `src/compact_wire.py` | 主响应的实际字节序列编码与解码 |
| `experiments/run_compact_revision.py` | 参数、攻击、逐步消融及审计实验 |
| `baselines/vgann/reproduction_v3/` | V-GANN 复现、正确性检查和共同查询对比 |
| `baselines/vsq/` | VSQ 集合版本检查、向量适配及留出集对比 |
| `evaluation_metadata/` | 固定语料与查询编号、归档输入校验信息 |
| `scripts/` | COCO 整理、向量编码输入、固定查询与基线准备 |
| `tests/` | 字段绑定、篡改、查询、排序、审计与传输格式测试 |
| `benchmarks/full_validation/` | 完整验证集的逐查询结果、重复计时与汇总 |

## 数据准备

仓库提供查询与语料的 COCO ID 映射，不分发 COCO 原始图像、预训练模型或大型向量文件。

1. 从 [COCO 官方网站](https://cocodataset.org/#download) 获取 2017 年训练集、验证集及 captions 标注，放入 `data/raw/coco/`，目录中应有 `train2017/`、`val2017/`、`annotations/`。
2. 按 [ImageBind 官方仓库](https://github.com/facebookresearch/ImageBind) 配置模型环境，并确保 `imagebind` 包可导入。编码另需 `pandas`、`pyarrow`、`tqdm`、PyTorch 及 ImageBind 的依赖。
3. 在仓库根目录执行：

```bash
python scripts/prepare_coco.py
python src/encoding/batch_encode.py
python scripts/generate_fixed_query_set.py
python scripts/prepare_benchmark_inputs.py
```

最后一步按照公开 ID 顺序生成 `results/evaluation_manifest_v1/`，包括冻结向量、查询及精确 Top-20 参照结果，同时生成 HCNNG 所需的 `fvecs` 文件。精确参照采用 float64 点积、`1e-8` 量化、记录编号升序打破同分规则。

`evaluation_metadata/gt_manifest.json` 保存论文实验输入的校验值。准备脚本输出 `input_verification.json`：使用原始归档向量时可核对一致性；重新运行编码器得到的向量若不同，应作为一次新的复现实验，不与归档结果混用。冻结向量进入检索后不再次归一化。

## 运行实验

### TrustMR 参数、攻击与消融

```bash
python experiments/run_compact_revision.py \
  --n 118287 --queries 100 --verify-queries 20 \
  --attack-queries 20 --audit-queries 3 \
  --output results/compact_run
```

该脚本包含 `mmax=4/8`、有无范围过滤、17 类攻击、真实跳过核验步骤的消融，以及三个受限查询审计案例。范围实验使用构造的时间戳与来源标识。这里的核验计时从已解码对象开始，外部对比的计时则包含解码，两者分开报告。

### 三方法共同查询对比

以下流程在 Linux 上运行，需要 Git、GNU `time`、`patch` 和支持 OpenMP 的 `g++`。HCNNG 构图代码取自固定上游提交，随机种子与同分顺序修改见随附补丁。

```bash
python scripts/build_hcnng.py
python baselines/vgann/reproduction_v3/run.py
python baselines/vgann/reproduction_v3/reference_check.py
python baselines/vgann/reproduction_v3/validate.py
python baselines/vgann/reproduction_v3/trustmr_wrapper.py
python baselines/vsq/original/run.py
python baselines/vsq/vector/run.py
python baselines/vsq/trustmr_wrapper.py
```

V-GANN 的 `run.py` 按预先确定的 R1/R2/R3 修复顺序执行正确性门槛检查，再运行共同查询。归档实验选中 R2。VSQ 在四组预定义参数上进行筛选，再固定配置进入留出集。文件名中的 R2 是复现版本标记，原论文方法名称为 V-GANN 和 VSQ。详见 [基线与计量说明](docs/REPRODUCTION.md)。

结果写入 `results/vgann_final_screen_v1/`、`results/vsq_original_v1/` 和 `results/vsq_vector_v1/`。完整实验计算量较大。使用新的结果目录运行，已有结果先另行归档，不在同一输出目录重复追加。

### 完整验证集评测

`run_full_validation.py` 将查询扩展至 COCO 验证集的全部 5,000 条图像和 5,000 条文本。检索库保持 118,287 张训练集图像，三种方法复用既有索引和固定配置。原有查询顺序与向量保留，新查询追加在后；每类前 100 条仍标记为配置筛选集，其余 4,900 条单独统计留出结果。

该脚本读取上述流程生成的索引、缓存及历史结果。输出目录必须位于原实验项目之外：

```bash
python experiments/run_full_validation.py \
  --archive "$PWD" --output ../trustmr_full_validation \
  --workers 16 --latency-queries 200 --latency-repeats 3
```

所有查询均执行实际响应编码和完整验证。并行阶段统计召回、响应字节数及验证是否通过；单查询延迟另取每类固定的 200 条留出查询，顺序执行 3 轮。重复计时不增加独立查询数量。`summary.csv` 分别汇总全验证集、留出集和筛选集，验证失败的数量单独列出。重启同一命令可跳过已保存的查询，进度见 `status.json`；全部步骤完成后生成 `complete.json`。

## 完整验证集结果

三种方法均完成 5,000 条图像和 5,000 条文本查询，检索库为 118,287 张训练集图像。下表报告全验证集均值，通信量为结果与证明的实际编码字节数。

| 方法 | 图像 Recall@20 | 文本 Recall@20 | 图像响应 (MiB) | 文本响应 (MiB) |
| --- | ---: | ---: | ---: | ---: |
| TrustMR | 0.9614 | 0.6983 | 27.838 | 27.838 |
| V-GANN | 0.2758 | 0.1705 | 0.763 | 0.782 |
| VSQ | 0.9434 | 0.4560 | 177.920 | 63.670 |

每类 4,900 条未参与配置筛选的留出查询单独报告在 `summary.csv` 中。完整响应验证的图像/文本均值分别为：TrustMR 3551.2/3534.6 ms，V-GANN 9.4/10.3 ms，VSQ 470.2/170.4 ms。计时采用每类 200 条留出查询、各重复 3 次的单工作进程测量。

原始结果及计量说明见 [完整验证集结果](benchmarks/full_validation/README.md)。参数、攻击、消融及随机审计实验采用各自配置和样本，不与完整验证集的主比较混合统计。

## 相关实现

- V-GANN：Wang et al., *Verifiable Graph-Based Approximate Nearest Neighbor Search*, ADMA 2024 proceedings, 2025. [DOI](https://doi.org/10.1007/978-981-96-0821-8_1)
- VSQ：Yin et al., *VSQ: Enabling efficient and verifiable similarity queries in blockchain databases*, Expert Systems with Applications, 285, 127815, 2025. [DOI](https://doi.org/10.1016/j.eswa.2025.127815)
- HCNNG：[作者构图实现](https://github.com/jalvarm/hcnng)，固定提交 `0fab25eb61b8852f4b4d3d4c05e0c9929647c96a`。
- ImageBind：[官方编码器实现](https://github.com/facebookresearch/ImageBind)。

本仓库的 V-GANN 与 VSQ 为依据论文原理构建的实验实现，非原作者官方实现。第三方模型、数据与代码遵循各自的许可条款。
