# Stage 0/1 概率预测接口

## 范围和状态

本轮实现 Stage 0 的概率接口、模型适配、评分与验收工具，以及 Stage 1 的 CPU 画像工具。真实 K/P 面板的 Stage 1 画像和 GPU 资源烟测是独立验收项；本地合成测试通过后仍需完成这两项，才可将完整 Stage 0/1 标为通过。B1-M/B1-Z/B1-F/B1-I 均须另行批准。

## 接口

- 新概率模型：`predict_quantiles(batch) -> [B,H,N,9]`，返回后处理前的 kW 分位数。
- `predict(batch)`：Q 模型返回统一下界截断后的 q0.5，旧点模型保留原行为。
- 新命令：`run-probability`、`score-probability`；与旧 `run`、`score` 的输出文件分离。
- 存储：`predictions_quantiles.npz` 包含 raw、scored、issue times、实体顺序、九个分位数水平与完整语义身份。旧评分器会拒绝四维概率文件。
- 输出目录必须为空，运行器不覆盖旧结果。文件身份包含任务哈希、panel/meta 哈希、归一化尺度、模型、seed 与后处理规则。
- `DONE` 对本概率运行的产物逐文件计算哈希；概率批跑的自动续跑尚未接入旧 study 管线。

## 固定协议

分位数：0.1、0.2、…、0.9。目标为现有 power，主 mask 为 observed。保存原始分位数后，按值升序重排并取 `max(q,0)`，保留高于额定容量的值。单调头出现交叉会直接报错。额定容量用作损失尺度。

主指标为 `normalized_mean_pinball`：对有效目标先索引，在容量归一化空间按有效格数×9取平均。训练损失使用原始容量归一化输出；验证选模和测试评分统一执行下界截断。检查点容差为 1e-4，容差内保留较早检查点。

辅助字段包含逐时距/逐分位数 Pinball、q0.5 MAE/RMSE、中心区间覆盖与宽度、严格/包含端点 CDF、WIS 分解及原始输出越界和交叉诊断。WIS 等于九分位数平均 Pinball 的两倍，二者共享同一证据。机组边际分位数不求和为全场概率区间。

## 模型清单

| 配置名 | 实现 | 说明 |
|---|---|---|
| pers_q | pers_q / conditioning=none | 训练残差按时距汇总 |
| pers_cq | pers_q / conditioning=level | 最新历史功率六档，样本不足回退 Pers-Q |
| pers_cq2 | pers_q / conditioning=level_volatility | 功率六档×训练波动三档；少于200回退，逐格记录 |
| dlinear_q / lstm_q / tcn_q / patchtst_q / itransformer_q / agcrn_q | 既有对应主干的 H×Q 输出 | 原有输入布局与缺失处理；统一非负累积增量，允许分位数相等 |
| lightgbm_q | 每个时距×分位数一个 booster | 24h摘要/滞后和当前水平；无日历相位、无实体ID；quantile目标 |
| toto1_zs_q | 官方 Toto1.0 原生混合分布 | 24分量 Student-t，单patch CDF二分，不使用本任务标签训练 |
| toto1_wpf_q | Toto1-WPF-Q | 原主干和原分组，容量归一化直接输出；head/last4/full |

现阶段实体布局Q训练器和池化LightGBM-Q要求机组额定容量一致，K/P满足2050kW条件。AGCRN-Q保留原AGCRN默认 batch，其余默认值同各自旧模型；loss统一为pinball。

Toto概率路径只接受明确的本地 checkpoint 目录，记录配置、权重及导入源码的哈希，不自动下载。主任务 H=36≤patch64，使用最终patch的前36个未来位置。H>64的自回归轨迹概率留待另立协议。源预训练是否见过K/P未逐来源排除，结果标明本任务标签使用情况。

## 时间协议

旧任务默认 `legacy` 不变。新增任务显式设置：

```json
{"extra":{"window_protocol":"causal_history_v1","probability_protocol":"unit_quantiles_v1"}}
```

- train 起报与历史均从训练分段内开始；train标签不越界。
- valid/test 起报属于各自分段，允许借用此前历史；标签全部留在该段。
- full训练起点沿用旧任务：K为2016-04-15，P为2016-09-01。
- 四个cutoff：2021-01-01、04-01、07-01、10-01；valid为cutoff前14天；test为下一季度。
- 输入包括起报行，标签为起报后1..H步。

## 安装与本地验收

使用 Python 3.12，与现有项目环境保持一致：

```bash
python -m pip install -e '.[deep,gbdt,test,toto,probability]'
python -m pytest tests -q
```

真实权重合成回归需要额外指定已有权重目录：

```bash
WPF_TOTO_DIR=/path/to/Toto1-checkpoint python -m pytest tests/test_toto_probability_real.py -q
```

未指定权重目录时该项明确跳过。其余测试不下载权重。

## Stage 1：CPU画像入口

以下命令读取已有面板，不重建数据、不加载模型、不生成预测；输出必须为新空目录：

```bash
PYTHONPATH=src python -m wpf_benchmark.profile_probability \
  --datasets /path/to/data/datasets \
  --out /path/to/task/stage1_profile
```

画像产出：profile、训练期波动边界、72组切窗审计、缺失/零值/超额定/分层计数、8份共同评分网格及分层mask。

- 训练波动边界按场站和cutoff估计，数据只来自训练历史。
- 测试数据仅参与计数，完整流程不读取模型结果。
- 小于200个issue×turbine的层保留并标记 `insufficient_n`。合并或改变边界需显式新版本。
- 测试网格的mask与finite标签决定评分格，与模型可用预测数无关。
- 诊断区分起报时可知条件和事后零功率/超额定/变化条件。
- 将 `extra.strata_boundaries_path` 指向真实冻结边界文件，后续概率运行会输出分层评分。

合成画像测试的产物带 `software_test_only=true`，正式概率运行拒绝将其用作真实边界。

## 配置生成入口

生成独立配置，过程不训练模型：

```bash
PYTHONPATH=src python -m wpf_benchmark.configure_probability \
  --datasets /path/to/data/datasets \
  --model-dir /path/to/Toto1-checkpoint \
  --out /path/to/task/configs
```

## K/P真实权重与资源烟测入口

下列命令包含规定范围内的短段训练和原生概率核对，须在用户授权的位置执行：

```bash
PYTHONPATH=src python -m wpf_benchmark.acceptance_probability \
  --datasets /path/to/data/datasets \
  --model-dir /path/to/Toto1-checkpoint \
  --device cuda \
  --out /path/to/task/toto_smoke
```

范围固定：双场×L144/576×head/last4/full，单轮、8个训练窗口、4个验证窗口、2个预测窗口；记录单轮耗时、峰值显存、参数量、冻结集合与时间边界。原生CDF与固定10000次采样、原均值在同一短段核对。不输出模型性能排名，不启动正式B1。

## 旧代码修复记录

GitHub拉取版本的 `models/toto.py` 存在相对导入层级错误、predict未局部导入torch、fit未设置ctx三项可执行性问题。本轮作最小修复，保留旧点路径的均值/采样中位数和额定裁剪语义。旧点评分器、旧模型配置均保持原样；既有神经checkpoint loader仅增加Q模型总参数计数兼容。

ClawsGO Science Agent
