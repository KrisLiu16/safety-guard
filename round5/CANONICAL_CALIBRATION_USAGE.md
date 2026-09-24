Canonical32 的校准与旧 bulk 评估相互独立。脚本必须等完整训练完成、最终选中权重及其 best_metrics 一致后执行；不重选权重，不读 fresh、旧官方或封存测试。

```sh
python /work/round5/calibrate_canonical_l20.py --calibrate \
  --runtime-code /work/round5/runtime \
  --training-output /work/output/round5/prefix_v2 \
  --output /work/output/round5/canonical32_calibration
```

默认 `--inference-engine eager`，每次物理 forward 固定 32 token，最后不足一块补未来 token，但仅统计真实输入位置且不提交部分块的缓存。也可显式选择 `window_cuda_graph`；实际引擎、初始化成本和源码均记录。只允许一张可见 L20，拒绝 CPU/M5 或静默降级。

每个初始/选中 checkpoint 各运行 cal 900、dev 1200；两份 SHA 相同时复用观测并注明。只用 cal 拟合语言/角色分层的 whole、stream 双阈值，均为严格 `>`。stream 是最后目标消息的所有原生 token 风险最大值与冻结的最多八个真实文本切点最大值；BPE 变化的切点重新执行，IDs 完全相同的原始前缀才复用概率。完整切点复用原端点，之前用户消息的风险不计为助手自然前缀标签。

选中权重的 canonical dev 门槛使用同执行方式的初始 window whole 宏召回为固定基准，要求下降不超过 2 个百分点且每层 stream FPR ≤5%。未通过仍保存产物，`canonical_quality_gate_pass=false`；进程的成功仅表示完整性与运行完成，不是质量通过。

输出为 `initial_window/` 和 `selected/` 下的 `calibration.json`、两份逐例预测、现代分词核对证明，以及根目录 `audit.json`。拒绝覆盖既有输出。中途失败保留已完成逐例和失败审计，不产生该阶段成功 calibration artifact。

真实分词对 frozen IDs 和原生目标边界逐条核对，不替换 IDs。五个共用执行源严格 SHA 绑定；实际实验 loader、HF/FLA 源另列。独立包内模型 loader 的重实现须通过独立 L20 包审计，不能由本校准宣称字节等价。禁止将固定轨迹经验 FPR 宣称为任意 BPE 到达切分的保证。

CPU 验证：`python3 -m unittest round5/test_calibrate_canonical_cpu.py -v`。当前只做代码、真实数据计数及 CPU 假模型反例测试，没有运行任何神经模型。

校准文件和两份权重身份锁定后，单独执行 fresh390：

```sh
python /work/round5/evaluate_canonical_fresh_l20.py --evaluate-fresh \
  --calibration-output /work/output/round5/canonical32_calibration \
  --output /work/output/round5/canonical32_fresh390
```

该入口不调用阈值拟合或选模函数。先检查校准产物 SHA、逐例预测 SHA 和固定阈值下的原 cal/dev 混淆矩阵，再验证 fresh 原始/切点现代分词，最后应用固定 whole/stream 阈值。fresh 390 条、390 个 family、六格各 65 条；历史 21 文件排除依据冻结清单与已完成的独立 CPU 审计，GPU 现场不重算历史全文件。文本排除仅覆盖规范化后长度至少 20 的字符串，短文本存在 26 个历史匹配，不能称所有长度零重叠。

fresh 两模型同 SHA 则复用观测；初始和选中权重各用其自己的已锁阈值。拒绝覆盖已有输出和未经校准的引擎替换。fresh 指标只作一次终测描述，不用于重新挑权重、修改阈值或改变门槛。CPU 验证：`python3 -m unittest round5/test_canonical_fresh_cpu.py -v`。
