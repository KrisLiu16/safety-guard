# Harbor 格式支持范围

**平台 Harbor 流程支持 Harbor 定义的全部 Task 格式，照钉住的 `harbor==0.23.0`（task.toml schema 1.4）执行。**
包括单步和多步（`[[steps]]`）、shared 和 separate 评估、skills、MCP 服务器、健康检查、Task 自带的已有上下文
（`trajectory.json`）、切换用户、collect 钩子、产物、reward.json 和 reward.txt、各阶段的网络要求。拦下的只有 Aster 的沙箱在物理上做不到的形态，列在最后，
导入时就作为 `blocking` 诊断标在 Task 上，不会到执行时才失败。

逐条对照的上游文件：

| 上游 | 预置流程里 |
| --- | --- |
| `models/task/config.py` | 字段、默认值（verifier 600 秒、collect 钩子 60 秒、健康检查的间隔和次数） |
| `models/task/verifier_mode.py` | shared 和 separate 怎么定，步骤的模式怎么继承 |
| `trial/single_step.py` | 单步：Agent → collect 钩子 → 取产物 → separate 时先停 Agent 环境 → 评估 |
| `trial/multi_step.py` | 多步：每一步的准备、评估、存档、停不停、分数怎么汇总 |
| `verifier/verifier.py` | tests 上传、test.sh 的执行、先读 reward.json 再读 reward.txt |
| `trial/artifact_handler.py` | 产物的布局；separate 时放回 verifier 沙箱里的原路径 |
| `trial/trial.py` 的 `_resolve_load_trajectory` | Task 自带的已有上下文在哪、什么时候灌进会话 |
| `environments/base.py` | 健康检查按 Docker HEALTHCHECK 的语义 |
| `agents/installed/claude_code.py` | Claude Code 的配置目录、skills、MCP、`--continue`、ATIF 转原生会话和 `--resume`、模型别名 |
| `agents/installed/codex.py` | Codex 的 `CODEX_HOME`、`config.toml`、skills、`exec resume --last`、ATIF 转 rollout |

Aster 的沙箱不是挂载型环境（和上游的 e2b、daytona 一样），上游按 `capabilities.mounted` 分支的地方都照非挂载那一支。
代码在 SDK 的 `aster_harbor` 包里，`aster docs show sdk-sources` 列出各模块的源码，每个函数的 docstring 写着它对应上游哪一段。

## 支持的能力

### Task 和 Agent

> 提示：创建 Run 时，建议将 Agent 时限设为「最长」（CLI：`--agent-timeout max`，也是默认值）。
> 模型网关负载高时，等待响应会占用运行时间，Task 设定的时限可能不足。使用最长时限可减少超时中断和重跑造成的资源浪费。

| task.toml | 平台怎么做 |
| --- | --- |
| `instruction.md` | 去掉开头的 canary 行，写到 `/logs/agent/instruction.md`，从标准输入交给 Agent；上限 200000 字符 |
| `[agent] timeout_sec` | Agent 时限。提交时默认是 `max`：每个 Task 取放得下的最大值（沙箱一次最多 24 小时，减去流程准备、评估、收尾和余量），这一项不管写没写都不用；给秒数同样每个 Task 都用这个数覆盖（等价上游自己的 `override_timeout_sec` 旋钮，只是 Aster 默认开着）。只有 `agent_timeout=task` 才照这项：上游可以不写（不限时），没写用提交时的兜底值 `resource_defaults.timeout_seconds`，没给兜底值或放不下沙箱的 24 小时上限整单被拒 |
| `[agent] user` | 以这个用户起 Agent；名字或 uid 都行，uid 用 `setpriv` 换，和 `docker exec -u` 一样 |
| `[environment] docker_image` | Agent 镜像；没写的在数据集里绑定一个构建好的镜像（平台不构建 `environment/Dockerfile`） |
| `[environment] workdir` | Agent 沙箱的工作目录；没写用镜像的 WORKDIR |
| `[environment] cpus`、`memory_mb` | 沙箱资源；内存按 GiB 申请，不整除时向上取整并给一条提示。上游可以不写（不限）；没写用提交时的兜底值 `resource_defaults.cpu`、`memory_gib`，没给兜底值整单被拒 |
| `[environment] storage_mb`（旧写法 `storage`） | 沙箱磁盘，和上游云上环境一样按它申请。沙箱只有 1、5、10、20、100 GiB 五档：向上取到最近一档（不整档给一条提示）；没写用提交时的兜底值 `resource_defaults.storage_gib`，不给是 20 GiB（上游不写是不限）。100 GiB 那一档要沙箱平台给账号开通，没开通时沙箱拿到的是平台默认的 1 GiB；separate 的 verifier 沙箱按它自己的环境算 |
| `[environment] env` | Agent 沙箱里每条命令的环境变量 |
| `[metadata]` | 原样给流程：`ctx.task.metadata` |
| `trajectory.json`；多步 Task 第一步的 `steps/<名字>/trajectory.json` | Task 自带的 ATIF 已有上下文。装 Agent 之前读出并查形状，不合法就是流程出错（`FLOW_SETUP_FAILED`）；第一次起 Agent 时灌进会话：Claude Code 转成原生会话放进配置目录用 `--resume <会话 ID>` 接着它，Codex 转成原生 rollout 放进 `$CODEX_HOME/sessions` 用 `exec resume --last` 接着它；其余脚手架照上游，不支持的在装之前拒绝 |
| `[task]`、`source`、`schema_version`、`version` | 接受；Task 身份和来源 |
| `build_timeout_sec`、`[solution]` | 接受但不使用：`build_timeout_sec` 只管构建镜像，`[solution]` 只给参考解（oracle）用，平台两样都不做 |

资源不够时（OOM、磁盘写满、超时）可以换资源重跑：`aster runs rerun <run> --samples <id> --resources-memory-gib 16 --resources-storage-gib 20`，
只改这几道 Rollout 自己的配置。

### Skills 和 MCP

| task.toml | Claude Code（`agent=claude-code`） | Codex（`agent=codex`） | 其余脚手架 |
| --- | --- | --- | --- |
| `[environment] skills_dir` | 每次起之前把目录里的内容拷进 `$CLAUDE_CONFIG_DIR/skills`（上游如此），镜像里 `~/.claude/skills` 也一起拷 | 拷进 `$HOME/.agents/skills` | 照上游拷进各家自己的 skills 目录 |
| `[[environment.mcp_servers]]` | 写进 `$CLAUDE_CONFIG_DIR/.claude.json` 的 `mcpServers`；`sse`、`streamable-http`（Claude Code 里叫 `http`）、`stdio` 三种都支持 | 写进 `config.toml` 的 `[mcp_servers.<名字>]`：`stdio` 写 `command`/`args`，其余写 `url` | 照上游写进各家自己的配置；opencode、mimo 的远程服务器写 `oauth: false`；mini-swe-agent 写成一段说明接在 instruction 后面 |

aider、trae-agent、swe-agent、nemo-agent、deerflow 不处理这两项，和钉住的上游一样不拒绝，Task 里的声明对它们不起作用。
各家的具体写法见[脚手架说明](harbor-scaffolds.md)。自定义流程用 `harbor.agent_input(spec, sandbox)` 把这两项交给适配器。

MCP 服务器是 Task 镜像里的进程或 Task 声明的地址；平台不另起 sidecar 容器。流程自己要起的后台服务用 `sandbox.start_service`。

### 健康检查

- `[environment.healthcheck]`：装 Agent 之前在 Agent 沙箱里跑，以默认用户执行（上游 `Trial._prepare`）。
- `[steps.healthcheck]`：多步 Task 每一步的 `setup.sh` 之后跑，以这一步的 Agent 用户执行。
- 语义照 Docker HEALTHCHECK：`start_period_sec` 内的失败不计数、按 `start_interval_sec` 重试；之后连续失败 `retries` 次就失败，
  两次之间隔 `interval_sec`，每次最多 `timeout_sec`。默认值和上游一样：5、30、0、5、3。
- 结果在时间线上是 `harbor.healthcheck` 节点。

### 评估

| 形态 | 平台怎么做 |
| --- | --- |
| shared（默认） | 在 Agent 沙箱里评估：Task 的 `tests/` 拷到 `/tests`，root 先 `chmod +x`，再以 `[verifier] user` 执行 `/tests/test.sh`，输出进 `/logs/verifier/test-stdout.txt` |
| separate（`[verifier] environment_mode = "separate"` 或写了 `[verifier.environment]`） | 先取回产物、停掉 Agent 沙箱，再按 Task 绑定的 verifier 镜像另起沙箱，放回产物，执行镜像自带的 `/tests/test.sh` |
| `[verifier] timeout_sec` | 评估时限，默认 600 秒 |
| `[verifier] env` | test.sh 的环境变量；separate 时叠在 `[verifier.environment] env`（没有就是 `[environment] env`）上 |
| `[[verifier.collect]]` | Agent 结束后在 Agent 沙箱里跑的命令（默认 60 秒，可以指定 `user`），失败只记下来不中止；时间线上是 `harbor.collect` |
| reward | `reward.json` 在就只看它，否则看 `reward.txt`；两个都没有是 `reward_missing`，空文件 `reward_empty`，解析不了 `reward_unparsable` |
| 没有评估声明 | 不评估；Agent 超时是 `AGENT_TIMEOUT`，非零退出是 `AGENT_FAILED`，被内核杀掉（137）是 `AGENT_OUT_OF_MEMORY` |

Agent 超时照样评估，和上游一样，Rollout 状态是 `timed_out`（超时），分数照常计入。Agent 自己以非零码退出和上游不同：上游照样评估、多步照样进下一步，
平台到此为止，结果码 `AGENT_FAILED`（退出码 137 是沙箱内存打满被内核杀掉，记 `AGENT_OUT_OF_MEMORY`），不评估、不进下一步。平台还会查分数：每个值是有限数、有 `reward` 键、`reward` 不是负数，
否则记 `reward_non_finite`、`reward_missing_key`、`reward_negative`。上游没有的 `verifier.fallback_reward` 不采用：
评估没出分就是 `no_reward`，原因留在结果里。

### 多步 Task

| task.toml | 平台怎么做 |
| --- | --- |
| `[[steps]] name` | 每一步的 instruction 是 `steps/<名字>/instruction.md`，必须在 |
| `steps/<名字>/workdir/` | 这一步开始前拷进工作目录；里面的 `setup.sh` 在拷完之后执行，失败就停在这一步 |
| `steps/<名字>/tests/` | shared 评估时先拷 `tests/` 再拷它，同名文件以步骤的为准 |
| `[steps.agent] timeout_sec`、`user` | 照 Task 时：这一步的 Agent 时限和用户，没写用 Task 的。提交时覆盖了 Agent 时限（`max` 或秒数），这一项不再用：所有步骤共用整个 Task 的预算，每一步最多用到剩下的时间；`user` 不受影响。和上游不同：上游把每一步的时限都换成覆盖值，三步 Task 就是三倍 |
| `[steps.verifier] timeout_sec`、`env`、`user`、`environment_mode` | 这一步的评估；时限没写是 600 秒，env 叠在 Task 的 `[verifier] env` 上，模式没写用 Task 的 |
| `[steps] min_reward` | 一个数（比 `reward`）或键到数的表；不到就停，缺的键按负无穷算 |
| `[[steps.artifacts]]` | 和 Task 的产物合在一起收 |
| `multi_step_reward_strategy` | `mean`（默认）：每个键在有分数的步骤上平均，缺的键按 0；`final`：最后一步的分数 |
| 会话接续 | [跨步骤继续会话](config-harbor-resume-between-steps.md)控制第二步起是否续接（上游 `agent.resume_trajectory`，Claude Code 用 `--continue`，Codex 用 `exec resume --last`），不清空 `/logs/agent` |

每一步开始前清空 `/logs/agent`（接续会话时不清），shared 评估前清空 `/logs/verifier` 和 `/tests`。
shared 评估一结束，就把 `/logs/verifier` 取到 Worker，并清空它和 `/tests`，下一步的 Agent 读不到上一步的测试和分数。
这些评估输出等最后一步的 Agent 结束后，再放进 `/logs/steps/<名字>/verifier`。
出错且没有评估结果时停下。separate 模式下 Agent 沙箱在最后一步评估前才停，每一步的 verifier 沙箱用完就停。
每一步的存档在 `/logs/steps/<名字>/{verifier,artifacts}`。下一步接着同一个会话时，起 Agent 之前把会话输出存进上一步的
`/logs/steps/<名字>/agent`：这次运行会覆盖会话里的输出日志。其余时候 Agent 的输出只在会话目录。时间线上每步有开始和结束两个 `harbor.step` 节点，
名字带上步骤和结果，比如 `harbor.step 1/2 one 完成 · reward 1`。

### 产物

- Task 的 `artifacts`：字符串（沙箱里的绝对路径）或 `{source, destination, exclude}`。
- 约定目录 `/logs/artifacts` 总会收，没写也补上。
- `/logs/agent` 只在沙箱里给 collect 钩子和测试读，不收回；Agent 的会话输出按会话目录收回（`sandboxes/<沙箱>/tmp/agent-session-<会话>/output/`）。
- 单步 Task 约定目录以外的产物按上游的本机布局放在 `/logs/harbor/artifacts`（有 `destination` 用它，否则是源路径去掉开头的 `/`）。
- separate 评估前取回 Worker，再放回 verifier 沙箱里的原路径；取不到的只记一条 `HARBOR_ARTIFACT` 警告。

### 网络

- 各阶段的 `network_mode`（`public`、`no-network`、`allowlist`）按 Task 冻结成 Agent 和 verifier 的要求。
- 哪些网络要求能提交到哪个放置组、哪种接入方式支持哪些要求，见[放置组](config-run-placement.md)和[边车转发](config-run-model-access.md)。
- setup、健康检查、Agent 和 collect 钩子在 Agent 阶段执行；shared/separate 评估使用 verifier 阶段，模型入口关闭。
- 同阶段保留本地服务；阶段切换回收旧进程与连接。平台装载/文件操作留在根侧，Task 网络策略不代表整个容器断网。
- 网络自证在受限阶段执行前完成；公网放置组根侧须可达，受限侧按策略做正负对照。
  不一致记 `SANDBOX_NETWORK_MISMATCH`，位置和阶段写进 `network_attested`。

## 导入时拦下的形态

这些 Task 在 `aster datasets version <version_id>` 的 `diagnostics` 里标为 `blocking`，不论流程声明什么都提交不了：

| 形态 | 原因 |
| --- | --- |
| `environment/docker-compose.yaml`（没写 `docker_image`） | 一个 Task 只有一个主容器，不支持多容器编排 |
| `artifacts` 或 `[[verifier.collect]]` 写了 `service` | 没有 sidecar 服务 |
| `gpus`、`gpu_types`、`tpu` | 沙箱没有 GPU 和 TPU |
| `storage_mb` 超过 102400 | 沙箱磁盘最大 100 GiB |
| `[environment] os = "windows"` | 沙箱只有 Linux |
| env 里引用宿主机变量（`${VAR}`） | 这是秘密的通道，要走平台凭证，本期不接 |
| 步骤级 `[steps.verifier.environment]` | 按 Task 冻结一个 verifier 镜像 |
| 步骤要 separate 评估而 Task 不是 separate | Task 没有 verifier 镜像 |
| 步骤级 `network_mode`、`allowed_hosts` | 网络要求按 Task 冻结 |
| 执行相关的表里有上游没有的字段 | 不知道它的执行语义 |
| 产物写相对路径 | 只收集绝对路径 |
| instruction 超过 200000 字符 | 上限 |

只提示、照样能跑的：没写 `cpus`、`memory_mb`、`storage_mb`、`[agent] timeout_sec`（提交时用兜底值）、`verifier.fallback_reward`（不采用）、`memory_mb` 不是整 GiB（向上取整）、`storage_mb` 不是整档（向上取到最近一档）。只给了 `environment/Dockerfile` 或 `tests/Dockerfile`、没写镜像的角色不另出诊断：缺镜像是那个角色的 `status: missing`，用得上它的流程提交时整单被拒，需要本机构建、推送、绑定，见[从数据集跑 Run](datasets-and-runs.md)。

`allowlist` 和 `allowed_hosts` 可导入，运行要求 gateway。域名只用于 DNS 与 TLS SNI；明文连接按 IP/CIDR 判断。
运行前自证的首个允许目标必须可探测；只有通配域名的清单不能提供正对照。真实 AGS 的 allowlist 整个 Task 验收尚未完成。

## 与 Harbor 的差异

- 多步 Task 的 shared 评估之后清空 `/tests` 和 `/logs/verifier`，下一步的 Agent 读不到上一步的测试和分数。钉住的版本还没有这一步，
  上游 main 在它之后修了同一个问题（`4008e2df`）。
- 步骤的评估时限：上游代码里有回落到 Task 时限的分支，但步骤的 `VerifierConfig.timeout_sec` 默认就是 600，那个分支走不到。
  Aster 照实际行为：步骤写了用步骤的，没写是 600 秒。
- Claude Code 的 key 用 `ANTHROPIC_AUTH_TOKEN`（平台的模型网关按 Bearer 认证），模型配置里的额外 header 经
  `ANTHROPIC_CUSTOM_HEADERS` 带上。
- 把上下文窗口告诉脚手架，上游不告诉。窗口写在 Harbor 模板顶上的 `context_window`（默认 `1_000_000`），
  不在表单里——要换值就直接编辑 `flow.py`，写 `None` 表示不写、由脚手架自己的默认接手。能配置的五家按各自的
  写法落下去：Claude Code 是 `CLAUDE_CODE_MAX_CONTEXT_TOKENS`，Codex 是 `config.toml` 顶上的
  `model_context_window`，kimi-cli 和 kimi-code 是 `max_context_size`（kimi-code 走变量那条路时是
  `KIMI_MODEL_MAX_CONTEXT_SIZE`），mcode 是 `config.yaml` 里这个模型下面的 `limit.context`（照上游的写法插，
  插不上这个 Rollout 就失败，不退回 mcode 的默认）。其余脚手架忽略。不给的话脚手架按自己的内置默认猜：猜小了白扔上下文，
  猜大了要等模型端拒了才知道，那时连压缩请求都超长，Agent 只能以非零码退出，这次 attempt 记 `AGENT_FAILED`、
  不评估。要用 Anthropic 的 1M 上下文，还要在模型配置的模型名后面写 `[1m]`（如 `claude-sonnet-5[1m]`）；
  自托管模型实际能承受的窗口是 sglang 启动时的 `--context-length`。
- 其余接 Anthropic Messages 的 Agent 也把 key 以 `Authorization: Bearer` 发（上游多数发 `x-api-key`），各家的写法见
  [脚手架说明](harbor-scaffolds.md)。只会发 `x-api-key` 的几家经网关时照常接 Anthropic Messages，直连时在装之前
  拒绝，错误里写明原因。
- Codex 只接 OpenAI Responses 协议的模型配置。连模型用自定义 provider（`[model_providers.aster]`），模型配置的额外 header
  写进它的 `http_headers`；上游写 `openai_base_url` 走内置的 openai provider，每次先连 7 次 WebSocket，也带不了 header。
- mini-swe-agent 的 `reasoning_effort` 按模型配置的协议传：Anthropic Messages 照上游写进 `model_kwargs.reasoning_effort`，并声明这个模型支持推理，
  否则 litellm 会丢掉它；Responses 照上游用 `litellm_response` 类，强度写 `model_kwargs.reasoning.effort`；Chat Completions 写进 `extra_body`
  （上游对 `openai/` 前缀的模型改打 `/v1/responses`，平台的网关不改写协议，Chat Completions 的模型配置会因此被拒）。
  上游的 `litellm_model_registry` 参数没有开：给了强度就一定声明。
- mini-swe-agent 在 Responses 协议下不给强度也用 `litellm_response`：litellm 的 `openai/responses/` 桥把「文字 + 函数调用」的响应拆成两个 choice，
  函数调用在第二个，默认类只读第一个，会报 No tool calls found。上游不给强度时打 Chat Completions，不经过桥。
- aider、swe-agent、openhands、openhands-sdk 不接 Responses 协议，与上游一致：上游只把 `provider/模型` 交给 litellm 或脚手架，没有 Responses 的写法。
  litellm 的 `openai/responses/` 桥把「文字 + 函数调用」的响应拆成两个 choice，这几家读第一个 choice，会丢掉函数调用。Responses 的模型配置在装之前就被拒。
- 平台适配器从脚手架仓库安装声明的精确版本，安装时核对冻结摘要和程序自报版本。
- 装 Agent 之前先对模型配置发一次 1 个 token 的请求：超时、连不上、5xx、429 是 `FLOW_SETUP_TEMPFAIL`（新建 attempt 重来），
  key 或模型名被拒是 `FLOW_SETUP_FAILED`；上游不探，网关不通时照样评估，得到 reward 0。自行执行命令的流程决定是否调用模型检查。
- Harbor 模板不构建镜像：缺镜像的 Task 先用[镜像构建模板](flow-template-image-build.md)构建推送，再绑定、发布版本。不跑参考解（oracle）。
