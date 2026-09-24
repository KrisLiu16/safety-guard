# 短卷积融合内核：待基线完成后评估的候选

2026-09-24，只读调查。**官方有具体匹配候选，但尚未在本项目 L20 上证明兼容或提速。** 当前先完成冻结的最终验证；本次没有下载轮子、安装依赖、运行模型或修改 `/work/modern`。以下仅是后续隔离实验的准入条件。

## 已核实的官方轮子

[Dao-AILab v1.7.0 官方发布](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0)，发布于 2026-08-20。官方 Release API 返回的资产：

- 文件：`causal_conv1d-1.7.0+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl`
- [固定版本下载地址](https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl)
- 大小：**196,353,608 bytes**。
- SHA-256：`6d04c5b9c675dd68aa4ece694d3c5d1d79b25326757bae567aeb2edfe1dd5cc2`。
- 元数据来源：[官方固定 tag API](https://api.github.com/repos/Dao-AILab/causal-conv1d/releases/tags/v1.7.0)。SHA 和大小来自发布方元数据，尚未下载后本地复算。

官方 [发布矩阵](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/.github/workflows/publish.yaml) 明确包含 Ubuntu 22.04 x86_64、Python 3.11、Torch **2.7.1**、CUDA toolkit **12.9.1**，且 Torch 2.7.1 排除 ABI=FALSE。其 [构建脚本](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/.github/workflows/_build.yml) 为 Torch 2.7 在该 toolkit 下选择 **cu128** 的 PyTorch wheel。因此它不是仅凭文件名猜测与 `2.7.1+cu128` 匹配。包的 [wheel 命名逻辑](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/setup.py) 使用 CUDA 主版本、Torch 主次版本和实际 C++ ABI；`cu12` 不能理解为“精确由 CUDA 12.8 nvcc 构建”。

## 本地环境证据与仍须确认的条件

现有记录确认 Torch `2.7.1+cu128`、Python **3.11**、Linux x86_64、Transformers `5.17.0`、FLA `0.5.2`、Triton `3.3.1`。Python/架构/Triton 可见于 [安装日志](../round3/l20/base_compare/output/install_modern.log)，Torch 与 L20 可见于 [训练日志](../round3/l20/base_compare/worker_final.log)；当前容器的 [编译工具安装日志](results/output/round4/compiler_install.log) 使用 Ubuntu jammy/amd64 源。记录未包含实际 `_GLIBCXX_USE_CXX11_ABI`，不能把官方默认值当作实测。

隔离实验前应只读核对：`sys.version`/SOABI 为 cp311；`platform.machine()` 为 x86_64；Torch/Transformers/FLA 版本不变；`torch._C._GLIBCXX_USE_CXX11_ABI is True`；`torch.version.cuda == '12.8'`；真实 L20 UUID、驱动与计算能力。随后检查 wheel 的标签、哈希及扩展 `.so` 的动态库/GLIBC/GLIBCXX 依赖，确认可在相同容器导入。该 wheel 标记为 `linux_x86_64`，不能按 manylinux 标签推断其全部系统兼容性。

若目标设备只读核实为 Ada/计算能力 8.9，官方构建源包含的 sm80/sm87 等 cubin 可依据 [NVIDIA Ada 兼容性说明](https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html) 的 Ampere→Ada 规则判断架构候选兼容性。这仍不是对这份 wheel 的实际加载、驱动兼容或数值正确性背书。

## 替换范围与备选

本地 [主干配置](../round3/l20/base_compare/output/qwen35_full/backbone_config.json) 是 18 个 GDN 层，每层短卷积 **6,144 channels、width=4、无 bias、SiLU**。官方 causal-conv1d 支持 depthwise conv、BF16 和 width 2/3/4；只替换实现，不需要改这些权重或分类头。[官方说明](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/README.md)

Transformers 5.17 的 [Qwen3.5 调用点](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py) 为：

- `causal_conv1d_fn(x, weight, bias, activation=...)`：`x=[B,D,T]`、`weight=[D,4]`。prefill 和多 token 追加由现有 cache 逻辑拼接历史后调用，再裁回新 token 部分。
- `causal_conv1d_update(x, conv_state, weight, bias, activation)`：单 token 且不记录历史时使用，必须原位更新整个短卷积状态，保持其地址与语义。

这些参数与 [Dao-AILab 接口](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/causal_conv1d/causal_conv1d_interface.py) 对应。Transformers 的 [dispatch 装饰器](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/integrations/hub_kernels.py) 在导入时解析原包；需在新进程、导入模型前提供依赖，并核验两个函数实际选择了融合实现。仅看到 `import causal_conv1d` 成功不足以证明模型已切换；也不能在已有审计进程中安装后期待自动生效。

**已有 FLA 0.5.2 的备选，不需新增二进制包：**[Triton causal_conv1d](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/modules/conv/causal_conv1d.py) 提供融合前向；[Triton update](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/modules/conv/triton/ops.py) 提供原位单步更新。但其输入是 `[B,T,D]`，返回 `(output, state)`；特别是 update 把输入视作各序列的**单步**，不能把 `T=8/32` 直接当作多步追加传进去。需要独立适配布局、返回值及 cache 语义，不能简单改函数名。优先调查官方匹配 wheel，FLA 作为 ABI/动态库不兼容时的候选；两者均不承诺更快。

## 后续隔离与验收顺序

1. **先保留当前最终速度证据。** 不覆盖现有包、模型清单、阈值或任何已完成结果。
2. 另起相同镜像的串行 L20 实验，新建独立环境/输出目录，复制锁定的依赖清单与模型资产引用。取得固定 URL 的 wheel 并复算上述大小/哈希，以 `--no-deps` 安装到新环境；不向 `/work/modern` 安装，不让 pip 升级 Torch/Transformers/FLA。不匹配就停止该 wheel 路线，不现场修改冻结环境补救。使用独立缓存目录并记录实际源码/扩展来源。
3. 先做算子对照：真实权重与激活布局、BF16/FP32、T=1/8/32 及较长 prefill；检查输出、精确原位状态更新、短于卷积宽度的输入。官方 [测试](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/tests/test_causal_conv1d.py) 包含 dtype/布局与 state 检查，可作参考，但其 BF16 容差不能直接替代整模型风险验收。
4. 同一冻结候选做原实现/融合实现的整段与流式对照，覆盖两个角色、512/8K 历史、变长块、BPE 回滚及会话隔离；记录全部状态和概率差，沿用已有 0.03 概率误差门槛，并单独报告固定阈值判断变化。CUDA Graph 需重新捕获并检查实际 cache 地址、重复 replay 与恢复，不能沿用旧 graph 或只验收 eager。
5. 数值通过后才跑同一 UUID、同协议的 128-call ITPS/P50/P95 和原始延迟记录，另报预热、显存和文本层开销。收益可能受投影/GDN/launch 开销限制；若没有稳定收益，保留当前运行时即可。此次不实施上述实验。
