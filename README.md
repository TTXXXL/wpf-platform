# 风电功率预测平台

把数据集、预测任务、模型和评估分开。新增数据集只需写一个适配器，新增模型只需写一个注册类；任务用一个 JSON 定义，同一套评估器给所有模型打分，批跑支持续跑和打包回传。

## 现有内容

| 类别 | 内容 |
|---|---|
| 数据集 | `sdwpf`（KDD Cup 2022，134 台，245 天）、`kelmarsh`（6 台）、`penmanshiel`（14 台）、`synthetic`（测试用） |
| 任务 | `sdwpf_m1_legacy`、`sdwpf_m2_legacy`（旧 SDWPF 协议）；`kelmarsh_actual_6h`、`penmanshiel_actual_6h` 及对应的 `_2h`（实际出力）；`synthetic_6h` |
| 模型 | 持续法、季节持续法、气候平均、Ridge、LightGBM、LSTM、TCN、DLinear、PatchTST、iTransformer、AGCRN（含当前功率残差版） |
| 预留 | NWP 等未来已知输入的对齐函数（尚未接入任务） |

## 快速开始

```bash
python -m pip install -e '.[deep,gbdt,test]'
wpf-benchmark build sdwpf                      # 从 data/raw/sdwpf/ 的两份 CSV 清洗构建
wpf-benchmark tasks                            # 检查任务配置
wpf-benchmark run --task sdwpf_m1_legacy --model persistence
wpf-benchmark study --plan baselines_v1 --preflight
```

## 文档

| 内容 | 入口 |
|---|---|
| 接口约定：数据存储、任务、模型、评估、批跑 | [平台接口](docs/平台接口.md) |
| 安装、命令、新增数据集或模型 | [使用指南](docs/使用指南.md) |
| 各数据集的来源、变量、掩码和已核实的事实 | [数据集](docs/数据集.md) |
| 各模型的结构、默认参数和与原论文的差别 | [模型](docs/模型.md) |
| 指标定义与结果文件 | [评估](docs/评估.md) |
| 服务器部署、批跑、续跑与回传 | [服务器部署与批跑](docs/服务器部署与批跑.md) |

旧版 SDWPF 研究代码（PIN、尾流、低功率头、固定预算双掩码、论文图件）保留在服务器版 Git 标签 `legacy-sdwpf-20261103`。新平台在旧切分和 M1 掩码下，持续法、Ridge 的全部指标与旧代码一致，相对差 ≤ 1e-11。

第三方模型参考代码的来源和许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
