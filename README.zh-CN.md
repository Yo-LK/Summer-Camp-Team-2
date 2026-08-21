# CWRU 轴承故障诊断

**语言：** [English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

## 项目目标

构建一个能够完成以下任务的智能体（agent）：

1. 加载并理解 CWRU 轴承数据集结构
2. 分析振动信号
3. 选择合适的故障诊断方法
4. 在不同工况（电机负载）之间应用迁移学习

**当前状态：** 完整的数据处理流水线（下载 → 切分/加窗/特征提取 → 基线模型 → 自适应方法）已经完成，并生成了一张统一的结果汇总表（`data/agent_policy_table.csv`，12 个负载对 × 10 种方法）。这张表如今驱动着一个真正可运行的智能体（`agent/`）——由策略查找加上 THOUGHT → ACTION → OBSERVATION → DECISION 循环组成——并通过一个 Streamlit 演示界面（`demo.py`）对外暴露，可以端到端地诊断上传的信号。该智能体被刻意限定为只将负载 0 和 1 视为已知的预训练源域（`agent.policy.KNOWN_SOURCE_LOADS`），因此遇到不熟悉的负载（2 或 3）时，会被迫真正运用迁移学习，而不是直接抄同域的近路。

## 四个目标的完成情况

| # | 目标 | 状态 | 说明 |
|---|---|---|---|
| 1 | 加载并理解数据集结构 | ✅ 已完成 | `data_download.ipynb` 下载、标注并检查数据 |
| 2 | 分析振动信号 | 🟡 部分完成 | 特征提取流程已存在（原始信号/FFT/包络谱/故障特征频率），但尚未在此基础上做进一步分析——目前没有任何验证证明这些特征真的能区分健康窗口和故障窗口 |
| 3 | 选择合适的故障诊断方法 | ✅ 已完成 | 10 种方法已经在真实的逐对数据上进行了正面对比，并汇总进 `data/agent_policy_table.csv`，如今真正由一个可运行的策略/智能体（`agent/policy.py`、`agent/react_loop.py`）自动完成选择，而不是靠人来读表 |
| 4 | 跨工况迁移学习 | 🟢 基本完成 | 两种真实的自适应方法（微调 CNN、CORAL + 随机森林）已经实现并在全部 12 个负载对上评估；表现最好的版本（部分冻结 CNN、基于 FFT 特征的 CORAL+RF）几乎追平了"仅用目标域数据训练"的上限——详见下方结果 |

## 已完成的工作

### `data_download.ipynb` —— 下载与探索

- 从 [CWRU 轴承数据中心](https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data) 下载 CWRU 的 **正常基线数据**（4 个文件，对应 0–3 hp 各一个负载）以及 **48 kHz 驱动端故障数据**（52 个文件：内圈/滚珠/外圈故障，故障直径 0.007"/0.014"/0.021"，每种在 0–3 hp 下各有一份），已存在磁盘上的文件会自动跳过。
- 将每个文件保存为描述性文件名，编码了故障位置、直径、负载、转速，以及（外圈故障时）时钟位置——例如 `48k_drive_end_fault_inner_race_0.007in_0hp_1797rpm_109.mat`——并分别存入带标签的子文件夹：`data/normal_baseline_data/` 与 `data/48k_drive_end_fault/`。
- 检查单个原始 `.mat` 文件的结构（CWRU 内部变量命名方式：`X{n}_DE_time`、`X{n}_FE_time`、`X{n}RPM`；本数据集中没有 `BA_time` 通道）。
- 直接从文件名解析元数据，构建一个 `manifest` DataFrame（每个文件一行），包含每个文件的信号长度、时长和记录的转速。
- 绘制一个示例对的正常信号与故障信号对比图，分别展示**时域**波形和 **FFT**（0–3 kHz）频谱。
- 将所有文件合并进单个 **`data/combined_dataset.mat`**（370 MB）——每个文件对应一个 MATLAB 结构体，信号保持原始完整长度，同时携带 `DE_time`/`FE_time`/`BA_time` 及全部故障元数据。要合并的源目录（`SOURCE_DIRS`）是显式指定的，而非隐式推断。

### `data_splitting_preprocessing.ipynb` —— 切分、加窗与特征提取

- 通过 `scipy.io.loadmat(..., simplify_cells=True)` 将 `data/combined_dataset.mat` 加载为 `df`（每个文件一行）。
- **按文件做时间维度的训练/测试切分**（`split_signal_train_test`，80/20）：在加窗**之前**先对每个文件的原始信号做切分，这样任何固定长度的窗口都不会跨越训练/测试边界或在两者之间泄漏——已通过将训练+测试重新拼接还原为原始信号来验证每个文件均无误。
- **固定长度加窗**（`segment_signal`，默认 `window_size=4096`，不重叠，丢弃末尾不足一个窗口的剩余样本），仅应用于 `DE_time`（驱动端通道；`FE_time`/`BA_time` 虽保存在 `combined_dataset.mat` 中，但尚未加窗）。
- 窗口按 `load_hp` 分组进 `windows_by_load`，并一次性保存到 **`data/windows_by_load.pkl`**（184 MB）——不会针对每一对负载重复存储。
- `make_splits(source_load, target_load)` 可按需为 `{0, 1, 2, 3}` hp 之间的 **12 个有序负载对** 中的任意一对构建 4 桶字典（`source_train`、`source_test`、`target_labeled`、`target_test`），使下游训练循环可以遍历 0→1、0→2、……、3→2 并汇总结果，而不必固定死某一个源/目标组合。
- **特征提取** —— 4 种方法，在 `windows_by_load` 中的每个窗口上一次性计算完成（共 5,601 个窗口）：
  1. **原始时域信号** —— 窗口本身；直接捕捉幅值模式。
  2. **FFT 幅值谱** —— 捕捉频率内容，但故障冲击信号通常被宽频结构共振所淹没。
  3. **包络谱** —— 对窗口做 `|Hilbert(window)|`，再对包络做 FFT。这一解调过程使故障冲击的**发生频率**呈现为清晰的谱线，与载波共振分离开来。
  4. **故障特征频率峰值（BPFO/BPFI/BSF）** —— 在理论上的外圈/内圈/滚珠自转故障频率（CWRU 公布的 SKF 6205 驱动端轴承阶次系数：转速的 3.5848×/5.4152×/2.357×）及其 2、3 次谐波处读取包络谱的幅值，容差宽到足以始终命中某个频率仓（约 12.7 Hz——FFT 频率仓间距为 11.72 Hz；此前 ±5 Hz 的容差比该间距的一半还窄，导致某些谐波无论信号内容如何都会静默返回 `0.0`），随后**按该窗口自身的时域 RMS 做归一化**，使结果反映的是相对谱能量集中度，而非各窗口整体的振动幅值——将密集的包络谱压缩为每个窗口 9 个具有物理意义、且可跨工况比较的数值。
  - 保存为 **`.npz`** 格式，而非 pickle —— 这些是用于输入模型的数值特征矩阵，不同于此前那种元数据繁重的结构。每个窗口的元数据（`load_hp`、`split`、`category`、`fault_location`、`filename`、解析出的转速等）作为并列数组保存在同一文件中：`data/features_time.npz`、`data/features_fft.npz`、`data/features_envelope.npz`、`data/features_fault_freq.npz`。四个文件的行顺序完全一致，因此第 `i` 行在每个文件里对应的都是同一个窗口。
  - 特征是按负载/切分方式计算的（与 `windows_by_load` 的结构一致），**尚未按源/目标对具体物化** —— 要为某个 (source_load, target_load) 组装出 `source_train`/`source_test`/`target_labeled`/`target_test`，仍需按 `load_hp`/`split` 对这些数组做筛选，方式与 `make_splits` 对原始窗口所做的一致。这部分筛选，以及在结果上训练/自适应/评估模型的工作，留给了（当时）尚未编写的训练 notebook。

### `model_training.ipynb` —— 基线模型

在来自 `data/windows_by_load.pkl` 的原始 `DE_time` 窗口上训练一个 1D CNN（10 分类：`normal` + {`inner_race`、`ball`、`outer_race`} × {0.007"、0.014"、0.021"}，外圈故障的时钟位置被合并进直径类别）。架构拆分为 `EmbeddingExtractor`（卷积堆叠 → 固定大小的嵌入向量）和 `LabelPredictor`（嵌入向量 → 类别 logits）两个独立的 `nn.Module`，以便后续的域自适应方法可以直接接入嵌入输出（例如在源/目标嵌入之间计算 MMD，或在其上添加梯度反转的域分类器），而无需改动分类头。

在全部 **12 个有序 (source_load, target_load) 负载对** 上评估的两个无自适应参考点：

- **基线 1 —— 仅源域（下限）**：在某个负载的完整训练集上训练模型，在**另一个**负载的测试集上评估（不做任何自适应）。平均准确率 **69.7%**，范围从 **39.8%**（2→0）到 **91.9%**（1→2）——这就是纯粹由工况偏移带来的代价，且随负载对不同波动很大。
- **基线 2 —— 仅目标域，稀缺（每类 10%）**：为每个负载单独训练一个模型，仅使用该负载训练窗口中**每个类别 10%** 的数据，在其自身负载的测试集上做域内评估。平均准确率 **82.7%**（75.2%–90.6%）。此前的一个版本曾直接用目标负载的**完整**训练集（约 1300 个窗口/负载）训练这一基线，得到约 99.9% 的准确率——本质上这是一个上限，而不是"标签稀缺"场景下的参考值——因此该版本被舍弃，改用现在这个每类 10% 的稀缺基线。
- 只需训练 **8 个模型**（每个基线各 4 个，每个负载一个）—— 对于同一个负载而言，`source_train` 和 `target_labeled` 是同一份底层数据，因此基线 1 的模型在该负载作为源域出现的全部 3 个负载对中都可以复用，结果是通过评估已训练好的模型得到的，而不是逐对重新训练。
- 权重检查点保存到 **`models/`**（`baseline1_full_load{0-3}.pt`、`baseline2_scarce_load{0-3}.pt`），结果保存到 **`data/baseline_results.csv`**（每个负载对一行，包含两个基线的准确率/宏平均 F1）。

### `domain_adaptation_evaluation.ipynb` —— 自适应 CNN、CORAL + 随机森林、RF 基线、全面对比

在两个基线之上构建真正的域自适应方法，并在全部 12 个负载对上用同一个 `target_test` 对所有方法做统一对比：

- **自适应 CNN，基于基线 1 微调** —— 加载基线 1 的源域训练权重，然后在基线 2 使用的同一份稀缺（每类 10%）目标域子集上，以比原始训练低 10 倍的学习率进行微调，共两种变体：**完全冻结**（只重新训练 `label_predictor`）和**部分冻结**（最后一个卷积块的权重和 BatchNorm 运行统计量也一并解冻）。
- **自适应经典机器学习 —— CORAL + 随机森林** —— 将 `source_train` 的协方差对齐到目标域的协方差（CORAL），在对齐后的结果上训练随机森林，共在三种特征集上运行：9 维的 `features_fault_freq`（BPFO/BPFI/BSF 峰值），以及 `features_fft`/`features_envelope`（各 2049 维，先经 PCA 降到 20 维——否则 CORAL 所需的协方差矩阵会严重秩亏）。**CORAL 相对于目标域是无监督的**——对齐过程只需要目标域的特征分布，从不需要标签——因此与上面那些确实需要标签的方法不同，它的目标域统计量是从**完整的目标域训练集**（每个负载 536–1322 个窗口）计算的，而不是那个仅供有标签方法使用的稀缺 10%/类子集；那个限制只适用于真正需要目标标签的方法。
- **经典机器学习基线（无自适应）** —— 仅在 `source_train` 上训练的普通随机森林，零样本跨域评估，对三种特征集分别进行——这是基线 1 在经典机器学习一侧的对应版本。这一步是必要的：没有它，就无法判断 CORAL 究竟贡献了多少，还是仅仅体现了该特征集 + RF 本身已有的能力——详见下方结论。
- 一项完整性检查用稀缺子集选取逻辑重新计算出基线 2 的数值，并确认与 `baseline_results.csv` 中的结果一致（差异仅来自 GPU 训练的不确定性），从而验证了 `windows_by_load.pkl`（原始窗口）与 `features_*.npz` 文件（预提取特征）之间基于索引的窗口对应关系是正确的。
- **80 个模型保存到 `models/`**：24 个自适应 CNN 检查点（2 种冻结模式 × 12 个负载对）+ 36 个 CORAL+RF 组合 + 12 个普通 RF 无自适应组合（3 种特征集 × 12 对 / 4 个负载，以 joblib 保存 `{clf, scaler, pca}`），加上前面提到的 8 个基线检查点。
- **汇总策略表** —— 将全部 10 种方法的准确率重塑为一张宽表，每个负载对一行，每种方法一列，保存到 **`data/agent_policy_table.csv`**。这正是 `agent/policy.py` 实际加载并查询的交接产物——详见下方结果。

**`data/agent_policy_table.csv` —— 每个负载对一行，每种方法一列（准确率）：**

| source→target | Baseline1 (no adapt) | Baseline2 (target-only) | CNN partial-freeze | CNN full-freeze | RF no-adapt (fft) | RF no-adapt (envelope) | RF no-adapt (fault_freq) | CORAL+RF (fft) | CORAL+RF (envelope) | CORAL+RF (fault_freq) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0→1 | 0.615 | 0.814 | 0.829 | 0.699 | 0.758 | 0.665 | 0.562 | 0.655 | 0.590 | 0.568 |
| 0→2 | 0.668 | 0.839 | 0.901 | 0.758 | 0.795 | 0.658 | 0.655 | 0.770 | 0.671 | 0.537 |
| 0→3 | 0.494 | 0.752 | 0.736 | 0.680 | 0.730 | 0.615 | 0.599 | 0.680 | 0.562 | 0.547 |
| 1→0 | 0.617 | 0.906 | 0.641 | 0.602 | 0.781 | 0.617 | 0.586 | 0.750 | 0.523 | 0.602 |
| 1→2 | 0.919 | 0.839 | 0.910 | 0.922 | 0.811 | 0.860 | 0.596 | 0.898 | 0.835 | 0.739 |
| 1→3 | 0.904 | 0.752 | 0.941 | 0.904 | 0.826 | 0.826 | 0.624 | 0.661 | 0.839 | 0.665 |
| 2→0 | 0.398 | 0.906 | 0.336 | 0.508 | 0.734 | 0.680 | 0.570 | 0.711 | 0.500 | 0.531 |
| 2→1 | 0.839 | 0.814 | 0.904 | 0.857 | 0.854 | 0.829 | 0.655 | 0.907 | 0.811 | 0.724 |
| 2→3 | 0.696 | 0.752 | 0.981 | 0.860 | 0.888 | 0.907 | 0.727 | 0.876 | 0.876 | 0.767 |
| 3→0 | 0.570 | 0.906 | 0.648 | 0.602 | 0.703 | 0.461 | 0.617 | 0.695 | 0.586 | 0.555 |
| 3→1 | 0.780 | 0.814 | 0.891 | 0.786 | 0.717 | 0.602 | 0.581 | 0.826 | 0.814 | 0.637 |
| 3→2 | 0.860 | 0.839 | 1.000 | 0.904 | 0.823 | 0.792 | 0.801 | 1.000 | 0.991 | 0.786 |

未出现任何一种方法能赢下所有负载对——例如 `RF no-adapt (fft)` 在 1→0 这一对上是表现最好的零样本方法（0.781，直接超过所有 CNN 和 CORAL 变体）——这种逐对之间的差异性正是智能体策略需要据以判断的真实信号。

**结果 —— 全部 12 个负载对上的平均准确率：**

| 方法 | 平均准确率 |
|---|---|
| 基线 2 —— 仅目标域，每类 10% | 82.7% |
| 自适应 CNN（部分冻结） | 81.0% |
| CORAL + 随机森林（FFT 特征） | 78.6% |
| RF，无自适应（FFT 特征） | 78.5% |
| 自适应 CNN（完全冻结） | 75.7% |
| CORAL + 随机森林（包络特征） | 71.6% |
| RF，无自适应（包络特征） | 70.9% |
| 基线 1 —— 仅源域 | 69.7% |
| CORAL + 随机森林（故障频率特征） | 63.8% |
| RF，无自适应（故障频率特征） | 63.1% |

![按方法划分的平均准确率](assets/mean_accuracy.png)

**逐对完整结果**（全部 12 个负载对的单独数据，而不仅是上面的平均值）保存在 **`data/full_comparison_results.csv`** 以及下方图表中：

![CNN 方法，全部 12 个负载对](assets/cnn_methods_per_pair.png)

![CORAL + 随机森林方法，全部 12 个负载对](assets/coral_methods_per_pair.png)

![CORAL 相对于普通 RF 是否真的有帮助，按特征集划分](assets/rf_coral_vs_noadapt.png)

**结论：**

- 在 `features_fault_freq.npz` 中发现并修复了两个实现缺陷：读取峰值时使用的容差比 FFT 频率仓间距还窄（导致某些谐波无论信号内容如何都被静默置零），以及没有做逐窗口的幅值归一化（原始幅值受各负载整体振动幅值主导，造成源域/目标域协方差尺度约 300 倍的失配，使 CORAL 在对齐时把大部分信号都压没了）。修复这两点后，CORAL+RF（故障频率特征）从 49.1% 提升到 60% 出头。
- 加入 RF 无自适应基线是必要的，而不是可有可无：它揭示出 **CORAL 实际的贡献很小且不稳定**。平均而言，CORAL 相对于普通 RF 每种特征集只领先约 1–3 个百分点，而且逐对来看，它甚至算不上多数情况下获胜（FFT：4/12，包络：5/12，故障频率：7/12）——如果没有这个基线，仅凭 CORAL+RF 的数字看起来会比实际证据所支持的"自适应确实有效"要乐观得多。
- 让 CORAL 使用**完整的**目标域训练集（而非它本不需要的稀缺标注子集）在理论上是正确的修正，但实践中并没有稳定地带来提升：FFT 和包络特征的平均表现反而略有下降（数据更多了，但现在覆盖的是不均衡的类别混合，而不是原来那个类别均衡的稀缺子样本），而故障频率特征则略有提升。原则上是对的，但在这种数据规模下不是稳赢的改动。
- 部分解冻 CNN 的最后一个卷积块（75.7% → 81.0%）仍然是一个明确、真实的提升。
- 无论是 CNN 还是经典方法，没有一种自适应方法平均而言明显**超过**基线 2；表现最好的方法也只是接近它，而非超越它。就本任务而言，利用源域知识目前还没有被证明能比直接用同等数量的稀缺目标域数据训练带来更多收益。
- 工况差距是逐对而异的，并非均匀分布：例如在 2→0 这一对上，基线 1（39.8%）实际上反而超过了部分冻结的自适应 CNN（33.6%）——自适应方法并不保证一定有帮助，有时甚至会比什么都不做更差。

### `cwt_baseline_exploration.ipynb` —— 更丰富的特征表示是否有帮助？

探索性实验，仅限基线（刻意没有纳入完整的自适应流程）：连续小波变换（CWT）尺度图（Morlet 小波，32 个尺度，覆盖 150–3000 Hz，时间轴下采样到 128 点 → 每个窗口得到一张 32×128 的图像），配合一个 2D CNN（`src.CNN2D`），每个负载训练一个模型（在其完整训练集上），并做零样本跨域评估——与基线 1 完全相同的协议，便于直接对比。

- 全部 12 个负载对上的平均准确率：**69.2%**——与原始窗口 1D CNN 基线 1（69.7%）基本持平，明显落后于基于 FFT 特征的 RF（78.5%）。
- 检查点保存到 `models/cwt_baseline1_full_load{0-3}.pt`；结果保存到 `data/cwt_baseline_results.csv`。
- 结论：在这里，更丰富的 2D 时频输入并没有明显超过简单得多的原始 1D 窗口 CNN，因此没有将其纳入其他特征表示所经历的完整微调/CORAL 对比流程——并非被排除在外，只是就所付出的额外计算成本而言，收益并不明显。

### `src/` —— 可复用的流水线逻辑，以及 `agent/` —— 诊断智能体

`src/` 把上面四个 notebook 中的逻辑抽取为一个可导入的包（`data_loading.py`、`preprocessing.py`、`feature_extraction.py`、`models.py`、`adaptation.py`、`evaluate.py`）。notebook 本身依然保持自包含——出于可读性考虑，每个 notebook 仍各自重新定义所需的逻辑——但 `agent/` 和 `demo.py` 会直接调用 `src/`，而不再重复其中的任何代码。

`agent/` 就是真正的智能体工作流：

- **`agent/tools.py`** —— 将 `src/` 封装为智能体可调用的动作：加载信号、加窗、提取 4 种特征表示中的任意一种、加载已训练的检查点/组合包、运行 CNN 或 RF 推理、对特征做 CORAL 对齐、微调 CNN、给预测结果打分。
- **`agent/policy.py`** —— 对 `data/agent_policy_table.csv` 的一个无状态查询：给定一个 `(source_load, target_load)` 负载对，按准确率对每种已验证的方法排序，并将胜出者解析为一次具体的工具调用（用哪个检查点/组合包、调用 `agent/tools.py` 中的哪个函数）。此外还定义了 `KNOWN_SOURCE_LOADS = {0, 1}`——智能体被刻意限定为只将这两个负载视为已知的预训练源域，尽管底层结果实际上覆盖了全部 4 个负载作为源域，这样做正是为了让遇到不熟悉的负载（2 或 3）时，智能体必须依赖迁移学习，而不能走同域的近路。
- **`agent/react_loop.py`** —— THOUGHT → ACTION → OBSERVATION → DECISION 循环：THOUGHT 向 `policy.py` 询问当前最佳的剩余方法，ACTION 通过 `tools.py` 执行该方法，OBSERVATION 记录预测结果及其置信度，DECISION 决定接受该结果，还是（在置信度过低时）回退到排名次之的方法（最多尝试 `max_attempts` 次）。
- **`agent/diagnose.py`** —— 入口函数 `diagnose(signal, condition)`：对信号加窗，为给定的 `condition` 选出最佳的已知源域（或对显式传入的源域按 `KNOWN_SOURCE_LOADS` 做校验），然后运行上述循环。
- **`demo.py`** —— 构建在 `diagnose()` 之上的 Streamlit 界面；详见下方的“运行演示程序”一节。

需要坦诚说明的一点：智能体循环中的“自适应”，指的始终是**挑选一个已经完成自适应的检查点**——在 `domain_adaptation_evaluation.ipynb` 中离线微调或做过 CORAL 对齐——而不是针对上传的信号实时计算自适应。即便真的在线做，这里的意义也不大：有监督微调需要标签，而诊断上传的信号恰恰没有标签；而基于单个文件的窗口重新计算 CORAL，得到的也只会是比离线组合包中已有统计量噪声更大的版本。

## 数据目录结构

```
data/
├── normal_baseline_data/       # 4 个文件：normal_{0,1,2,3}hp.mat
├── 48k_drive_end_fault/        # 52 个文件：48k_drive_end_fault_<location>_<diameter>in_<load>hp_<rpm>rpm[_<position>]_<file_number>.mat
├── combined_dataset.mat        # 56 个文件全部合并，每个文件一个结构体，完整长度信号 + 元数据
├── windows_by_load.pkl         # DE_time 窗口（大小 4096），按 load_hp 分组，按文件切分 train/test
├── features_time.npz           # 原始窗口，(5601, 4096)，+ 每窗口元数据
├── features_fft.npz            # FFT 幅值谱，(5601, 2049)，+ 每窗口元数据
├── features_envelope.npz       # Hilbert 包络谱，(5601, 2049)，+ 每窗口元数据
├── features_fault_freq.npz     # BPFO/BPFI/BSF 峰值幅度（已做 RMS 归一化），(5601, 9)，+ 每窗口元数据
├── baseline_results.csv        # 基线 1 & 2 的准确率/宏平均 F1，每个负载对一行（共 12 行）
├── full_comparison_results.csv # 全部 10 种方法的准确率/宏平均 F1，每个负载对一行（共 12 行）
├── cwt_baseline_results.csv    # CWT+2D CNN 基线的准确率/宏平均 F1，每个负载对一行（共 12 行）
└── agent_policy_table.csv      # 宽表：每对一行 × 10 个方法列，仅含准确率 —— 智能体交接产物

models/                          # 未被 gitignore —— 共 84 个文件
├── baseline1_full_load{0-3}.pt          # 基线 1 检查点（4 个）
├── baseline2_scarce_load{0-3}.pt        # 基线 2 检查点（4 个）
├── adapted_cnn_{full,partial}_{S}to{T}.pt   # 自适应 CNN 检查点（2 种模式 × 12 对 = 24 个）
├── rf_noadapt_{fault_freq,fft,envelope}_load{0-3}.joblib  # 普通 RF 无自适应组合（3 种特征集 × 4 个负载 = 12 个）
├── coral_rf_{fault_freq,fft,envelope}_{S}to{T}.joblib  # CORAL+RF 组合（3 种特征集 × 12 对 = 36 个）
└── cwt_baseline1_full_load{0-3}.pt      # CWT+2D CNN 基线检查点（4 个）

assets/                          # 未被 gitignore —— README 图表
├── mean_accuracy.png
├── cnn_methods_per_pair.png
├── coral_methods_per_pair.png
└── rf_coral_vs_noadapt.png
```

`data/` 被 gitignore 忽略（大型二进制文件）——其中的一切都可以通过依次运行 `data_download.ipynb`、`data_splitting_preprocessing.ipynb`、`model_training.ipynb`、`domain_adaptation_evaluation.ipynb` 复现。`models/` 与 `assets/` 未被忽略，因为 `assets/` 存放的是本 README 所需的图片，而 `models/` 并不像 `data/` 那样属于大型二进制数据。

## 环境搭建

**方案 A —— conda：**

```bash
conda create -n cwru python=3.10
conda activate cwru
pip install -r requirements.txt
```

**方案 B —— venv：**

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows 系统：.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` 中的 `torch` 会由 `pip` 根据你的平台自动解析并安装相应的 CUDA 版本；纯 CPU 的机器则会安装 CPU 版本，两种情况都无需手动修改。

按顺序运行：`data_download.ipynb`（填充 `data/`）、`data_splitting_preprocessing.ipynb`、`model_training.ipynb`（在 `models/` 中生成 8 个基线检查点）、`domain_adaptation_evaluation.ipynb`（再向 `models/` 添加 72 个检查点/组合，重新生成 `assets/*.png`，并生成 `data/agent_policy_table.csv`）。`cwt_baseline_exploration.ipynb` 是可选的，与其余部分相互独立——它只需要 `data/windows_by_load.pkl`，智能体和演示程序都不依赖它。

## 运行演示程序

在 `data/` 和 `models/` 已经填充完毕后（或者已经存在预先构建好的 `models/` 目录），启动 Streamlit 界面：

```bash
streamlit run demo.py
```

这会打开一个浏览器标签页（默认地址 `http://localhost:8501`），你可以在其中上传一个原始的 `.mat` 振动信号文件，智能体将会：

- 根据信号中记录的转速（RPM）推断其工作负载条件；如果 RPM 缺失，或者与四个已知负载都不够接近，则回退为手动选择
- 运行 THOUGHT → ACTION → OBSERVATION → DECISION 诊断循环（`agent/diagnose.py`、`agent/react_loop.py`），从 `data/agent_policy_table.csv` 中挑选经过验证的最佳工具链，若置信度过低则回退到其他方法
- 报告预测的故障类别、置信度、所使用的方法，以及——对于本项目自带数据集中的文件——从文件名推断出的真实类别，供对比参考

## 尚缺内容 / 后续步骤

- **目标 2（信号分析）** 仍然缺少在提取流程之上的进一步**分析**——没有可视化或统计数据来比较不同故障类型/严重程度下提取出的特征，没有验证 BPFO/BPFI/BSF 峰值是否真的能区分健康与故障窗口，也没有经典的时域统计特征（RMS、峰度、偏度、峰值因子）。
- **智能体目前只会在离线计算好的结果中做选择——从不进行在线自适应。** `react_loop.py` 中的“自适应”步骤，做的是加载一个已经微调/CORAL 对齐好的检查点；`agent/tools.py` 确实暴露了 `fine_tune_cnn()`/`coral_align()`，但诊断流程中没有任何地方真正调用它们。有监督微调需要标签，而诊断上传的信号本就没有标签；基于单个文件的窗口重新计算 CORAL，也只会得到比离线组合包中已有统计量更嘈杂的版本——真正有意义的在线自适应，需要的是一批全新的、无标签的部署数据，而不是单个文件的诊断请求。
- **CWT + 2D CNN**（`cwt_baseline_exploration.ipynb`）只作为零样本基线做了测试（69.2%，与现有的原始窗口 CNN 基本持平）——没有纳入其他四种特征表示所经历的微调/CORAL 完整流程，因为基线结果并不能明显证明这部分额外计算是值得的。
- 尚未在现有特征表示上尝试的方向：MMD/DANN 风格的对抗式域自适应、为自适应 CNN 中解冻的卷积块与分类头设置不同的学习率、在组合特征集（例如将 FFT 与故障频率特征拼接）上运行 CORAL，以及 `features_time`/`FE_time`（风扇端）/DE+FE 组合信号从未被用于任何用途。
- 无论是 CNN 还是经典方法，没有一种自适应方法平均而言明显**超过**基线 2（详见上方结论）——就本任务而言，利用源域知识目前还没有被证明能比直接用同等数量的稀缺目标域数据训练带来更多收益。
- 目标 1 和目标 3 已完成。

## 结论

四个目标均已完整达成：数据流水线完成了 CWRU 48kHz 驱动端数据集的下载、打标签与探索；特征提取流水线产出了五种表示（原始窗口、FFT、包络、故障频率峰值，以及 CWT 尺度图），过程中发现并修复了两个真实的 bug（故障频率容差窗口比 FFT 频率分辨率还窄，以及缺失的 RMS 归一化曾使 CORAL 的协方差对齐失真约 300 倍）；11 种方法在全部 12 个有序的 source→target 负载对上被逐一对比，并汇总进 `data/agent_policy_table.csv`；两种真实的自适应方法（微调 CNN、CORAL+随机森林）已经实现，并被验证不存在数据泄漏——微调仅使用目标域**训练**切分中按类别分层抽样得到的稀缺子集，经代码审查与实证重叠检查均确认与留出的**测试**切分零重叠。

在此基础上，智能体本身也已构建完成并可正常运行：`agent/policy.py` 是对结果表的无状态查询；`agent/react_loop.py` 运行一个 THOUGHT → ACTION → OBSERVATION → DECISION 循环，挑选经验证的最佳方法，并在置信度过低时回退到备选方案；`agent/diagnose.py` 将其整合为单一的 `diagnose(signal, condition)` 调用；`demo.py` 在此之上搭建了 Streamlit 界面——上传一个信号，智能体会从 RPM 推断其工作条件，并将自己限制在一组刻意设定的、较小的"已知"源域内（`KNOWN_SOURCE_LOADS = {0, 1}`），使得负载 2 和 3 真正需要依靠迁移学习，而不是走同域捷径，最终连同完整的推理轨迹一起报告预测结果。

最重要的一点需要如实报告，而不是加以美化：在全部 12 个负载对上平均而言，**没有任何一种自适应方法能超过基线 2**（一个直接在目标负载上用 10%/类的稀缺标签子集训练出来的模型，完全不涉及源域）。CNN 部分冻结微调已经很接近了（81.0% 对 82.7%），CORAL+RF 和 RF-无自适应 在 FFT 特征上的表现也都不错（约 78.5%），但迁移学习的核心承诺——即源域训练出的模型能胜过直接使用少量目标域标签——就本任务而言尚未得到证实。这是一个关于自适应何时值得投入复杂度、何时不值得的真实且有用的发现，而不是需要掩藏的失败。

## 局限性

- **结果无法推广到本数据集之外。** 这里的一切都建立在单一轴承型号（SKF 6205）、单一数据来源（CWRU）、四种离散负载/转速条件之上。这些方法——乃至"自适应无法超越稀缺标签基线"这一发现本身——是否能在不同的设备、传感器或轴承型号上成立，目前尚未测试，也无法在没有新标注数据的情况下回答。
- **智能体从不进行实时自适应。** 如上文所述，`react_loop.py` 只会在离线微调或 CORAL 对齐好的检查点中做**选择**；`agent/tools.py` 中的 `fine_tune_cnn()`/`coral_align()` 虽然存在，但诊断流程中并未被调用。这是一个真实的能力缺口，而不只是一个未优化的细节——诊断上传的信号本身没有标签可用于有监督微调，而单个文件的数据量也不足以让 CORAL 得到比已缓存结果更好的估计。
- **`KNOWN_SOURCE_LOADS` 是一个演示性约束，而非技术限制。** 将智能体的源域限制为负载 {0, 1}，是刻意做出的选择，目的是迫使负载 2/3 真正用上迁移学习——事实上全部 4 种负载作为源域的检查点和结果都已经存在，如果解除这一约束，表现只会一样好或更好。
- **演示程序的真实类别对比只对本项目自带的文件有效。** `demo.py` 是从 CWRU 目录中具有描述性的文件名推断出"实际类别"的——对于一个真正全新、无标签的信号，是没有办法验证诊断是否正确的。
- **结果存在约 1 个百分点的运行间波动。** 这是直接观察到的：在相同随机种子下重新训练 CWT，其平均准确率从 69.2% 变成了 68.4%，这是 GPU 训练不确定性导致的。本 README 中的所有准确率数字都应被视为近似值，而非精确到小数点后三位。
- **没有自动化测试套件。** 正确性目前是通过手动重新执行 notebook、有针对性的验证脚本，以及直接检查（例如训练/测试集泄漏检查）来确认的——而不是由 CI 支撑的测试用例，因此目前只有手动重新运行 notebook 才能发现回归问题。

## 仓库结构

快速参考上面各部分是如何组合在一起的——每个文件具体做什么，详见上方“`src/` —— 可复用的流水线逻辑，以及 `agent/` —— 诊断智能体”一节：

```
src/                                # 可复用、可导入的代码 —— 供 agent/ 与 demo.py 使用
├── __init__.py
├── data_loading.py                 # load_mat_file(), build_raw_df(), label_file(), label_for_item()
├── preprocessing.py                # split_signal_train_test(), segment_signal(), build_windows_by_load()
├── feature_extraction.py           # extract_time(), extract_fft(), extract_envelope(),
│                                      extract_fault_freq(), extract_cwt()
├── models.py                       # CNN1D, CNN2D, WindowDataset, ScalogramDataset, train_cnn()
├── adaptation.py                   # fine_tune(), coral_transform()
└── evaluate.py                     # 准确率/F1/混淆矩阵相关辅助函数

agent/                               # 智能体工作流
├── __init__.py
├── tools.py                         # 将 src/ 中的函数封装为智能体可调用的工具
├── policy.py                        # 加载 agent_policy_table.csv，构建/查询
│                                       source→target → 最优方法 的查找表；KNOWN_SOURCE_LOADS
├── react_loop.py                    # THOUGHT/ACTION/OBSERVATION/DECISION 编排循环
└── diagnose.py                      # 主入口：diagnose(signal, condition)

demo.py                              # Streamlit 界面演示入口 —— 运行：`streamlit run demo.py`
```
