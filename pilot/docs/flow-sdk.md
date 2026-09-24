# 流程 SDK 参考：`aster_flow`

在流程入口 `run(ctx)` 中使用以下接口。教程见[编写流程](flow-authoring.md)。

可组合的 Flow 资源与执行 SDK。Harbor 标准实践和 Agent 适配器各自在独立包中。

## 入口

### `flow(*, setup_timeout: int, finish_timeout: int, atif: str = 'none', images: dict[str, str] | None = None, limits: dict[str, int] | None = None, requires: Sequence[str] = ()) -> Callable[[F], F]`

声明 run(ctx) 的准备/收尾时限（秒）、轨迹转换方式和用到的 Task 配置。

两个时限必须显式填写，范围 1–86400。使用单个标准 Agent 时，平台从适配器
推导轨迹转换；自行生成轨迹可指定 atif="file"。声明由服务端静态读取，不执行源码。

`requires` 列出流程读的 Task 配置，取值 `instruction`、`agent_environment`、`evaluation`。
提交时只检查和冻结声明了的：没声明的 Task 配置不会拦下提交，`ctx.task` 上也读不到。
声明 `evaluation` 必须同时声明 `agent_environment`；没声明 `agent_environment` 时必须写 `limits`。

### `Context`

流程入口 run(ctx) 收到的上下文。

从 `task` 读取任务，用 `models[名称]` 选择模型，用 `sandboxes.create()` 创建沙箱。
`scaffolds` 安装指定版本的脚手架，`agent` 启动 Agent，`materials` 将素材复制到沙箱。
运行后通过 `artifacts` 收集文件、`results` 保存自定义结果、`evaluation` 提交评分。

`flow_files` 读取提交时保存的流程文件，`files` 读取本次用户目录。
`api` 以提交人的身份调用 Aster 接口，`retry` 用于重试临时故障。
沙箱需由流程创建，数量受流程的资源预算限制。

- `model: Model`：仅声明一个模型时取该模型；多个模型请使用 ctx.models[key]。

#### `retry(call: Callable[[], T], *, on: type[Exception] | tuple[type[Exception], ...], what: str) -> T`

调用 call，遇到 on 指定的异常时重试，日志中用 what 说明当前操作。

重试间隔由 Worker 设置，逐次增加，到达上限后保持不变。剩余时间不足时抛出原异常。
只对临时故障使用此方法；call 必须能重复执行，例如重试上传前要先复位文件读取位置。

#### `emit(name: str, data: Any = None) -> None`

在这次 Rollout 的时间线上加一个节点（事件种类 `flow`），页面按 `name` 显示，`data` 展开在详情里。

`name` 1–64 个字符；`data` 要能序列化成 JSON。每个 attempt 最多 500 个节点，每个的 `data` 序列化后
不超过 16 KiB，超出的平台丢掉并记一条 `FLOW_EVENT_LIMIT` 警告。值按这次 attempt 的秘密脱敏。

#### `report_scaffold(name: str, version: str) -> None`

装好的脚手架和它自报的版本，记成 `scaffold_ready`。页面按名字选 Agent 输出的解析器（`claude-code`）。

#### `warn(message: str, *, code: str = 'FLOW_WARNING') -> None`

记一条警告事件：不改变结果，页面上看得到。

## Task 和这次尝试

### `Task`

本次执行的 Task，包含数据集信息、文件及流程声明要使用的配置。

总是有的：

- `key`：task_key；`id`：数据集版本里这个 Task 那一行的 ID。
- `dataset`：`Dataset`。
- `metadata`：`task.toml` 的 `[metadata]`，或 JSON 导入时带的 metadata。
- `files`：Task 目录，`TaskFiles`；JSON 导入的 Task 没有目录，是 None。
- `tests`：评估测试目录，`files.sub("tests")`；Task 目录里没有 `tests/` 就是 None。
- `toml`：`task.toml` 原样解析成的字典。

流程在 `@flow(requires=...)` 里声明了才有，没声明时读取抛 `TaskConfigNotDeclared`：

- `instruction`（声明 `instruction`）：instruction 内容（Harbor Task 去掉了 canary 行）；多步 Task 是空串，每一步的 instruction 在
  `steps/<名字>/instruction.md`。`instruction_file`：平台已经把 instruction 写到的沙箱文件，`/logs/agent/instruction.md`。
- 声明 `agent_environment`：`environments["agent"]` 是 agent 沙箱的环境规格；`images["agent"]`；
  `workdir` 是 Task 声明的工作目录，实际创建后的环境目录见 `sandbox.workdir`；
  `resources` 是 `{"cpu": 核数, "memory_gib": GiB, "storage_gib": 磁盘 GiB}`；`timeouts["agent"]` 是 Agent 时限（秒）；
  `agent_timeout_override` 是提交时覆盖了 Task 自己的 Agent 时限时的那个秒数，照 Task 自己的时限是 None，
  多步 Task 覆盖时各步不再单独限时，共用这一个预算；`network` 是 `{"agent": ..., "verifier": ...}` 各阶段的网络要求，
  Task 没写是 None。
- 声明 `evaluation`：`verifier` 是冻结的评估声明（environment、timeout_seconds、env、collect、artifacts……），
  Task 不评估是 None；`timeouts["verifier"]` 是评估时限，0 表示不评估；separate 评估才有
  `environments["verifier"]` 和 `images["verifier"]`。

- `instruction: str`
- `instruction_file: str`
- `workdir: str`
- `resources: dict[str, int]`
- `agent_timeout_override: int | None`
- `network: dict[str, Any] | None`
- `verifier: dict[str, Any] | None`
- `toml: dict[str, Any]`

### `Dataset`

Task 所属的数据集版本。数据集之后被删，名字和版本号是 None。

- `version_id: str`
- `id: str | None`
- `name: str | None`
- `version_no: int | None`

### `Image`

一个沙箱的镜像：冻结的 `名字@sha256`，和镜像库给出的完整地址。

- `reference: str`
- `location: str`

### `Paths`

沙箱里的约定目录（上游 Harbor 的 EnvironmentPaths）。

- `agent_dir: str`：`/logs/agent`：Harbor 约定的 Agent 日志目录，Task 的 collect 钩子和测试从这里读。平台不自动收回：
Agent 会话的输出在会话自己的目录，平台按会话收回、转 ATIF。
- `verifier_dir: str`：`/logs/verifier`：评估脚本写 reward 和输出的地方。
- `artifacts_dir: str`：`/logs/artifacts`：约定的产物目录，separate 评估时要放进 verifier 沙箱。
- `tests_dir: str`：`/tests`：评估测试下发到这里。

### `Attempt`

这一次尝试：`run_id`、`run_no`、`sample_id`（Rollout）、`sample_no`、`attempt_id`、
`attempt_number`（这一轮里第几次尝试）。

#### `remaining_seconds() -> float`

流程时限（setup + Agent + 评估 + finish）大约还剩多少秒。等平台起沙箱的时间不算。

## 模型和 TITO

### `Model`

Run 中的一份模型配置，通过 ctx.models[名称] 获取。请求体使用该模型的原生协议。

#### `connect(accepts: list[str] | tuple[str, ...], *, sandbox: Sandbox, session_id: str | None = None) -> Endpoint`

返回沙箱可用的模型连接信息。

`accepts` 列出脚手架支持的协议。优先使用模型配置中的协议；开启 TITO 时，也可选择
TITO 支持的其他协议。没有共同协议时抛 ValueError。
gateway 模式下会确保沙箱中的 sidecar 已启动。预检发现后端或凭证不符合要求时抛
ModelRejected；连接故障或服务端错误抛 TempFail。

#### `check(timeout: float = 60.0, *, sandbox: Sandbox | None = None, retry_seconds: float = 600.0) -> ModelCheck`

检查模型是否可调用；传入 sandbox 时，通过该沙箱的模型入口检查。

一次探测最多等 `timeout` 秒。超时、连不上、429、5xx 按 Worker 给的间隔退避重探（SG-14），从第一次探测算起
`retry_seconds` 秒内（最后一次探测也在这之内结束）还不通就抛 `TempFail`；`retry_seconds` 不大于 `timeout`
时只探一次。其余 4xx 直接抛 `ModelRejected`，不重探。在 Agent 启动之前调用时，重探的时间算在 `setup_timeout` 里。

#### `request(body: dict[str, Any], *, timeout: float) -> dict[str, Any]`

直接请求模型；body 是该协议的原生 JSON，流式响应使用 stream。Gemini 打 `:generateContent`。

#### `arequest(body: dict[str, Any], *, timeout: float) -> dict[str, Any]`

异步原生请求，取消关闭该请求的连接与网关。

#### `stream(body: dict[str, Any], *, timeout: float) -> Iterator[bytes]`

返回响应字节流；OpenAI 和 Anthropic 由调用者在请求里显式设置 stream 字段，Gemini 打 `:streamGenerateContent?alt=sse`。

#### `astream(body: dict[str, Any], *, timeout: float) -> AsyncIterator[bytes]`

异步字节流，关闭迭代时回收这次请求。

### `ModelCheck`

探测通过：地址、状态码和耗时。

- `endpoint: str`
- `status: int`
- `seconds: float`

### `Endpoint`

供指定沙箱使用的模型地址、凭证、协议和请求头。

- `base_url: str`
- `api_key: str`
- `wire_api: str`
- `headers: dict[str, str]`

### `Tito`

冻结的采集开关；采集由模型 sidecar 的生命周期管理。

## 脚手架仓库

### `Scaffolds`

`ctx.scaffolds`：这次提交冻结了哪些脚手架版本，从打包件存储解进沙箱。

flow.py 里声明的 Agent 版本在提交时按平台登记的版本解析，冻结脚手架名、版本和各平台的树摘要、入口、运行时；
只有冻结了的版本能装。

- `frozen: list[tuple[str, str]]`：这次提交冻结了的 `(脚手架名, 版本)`。

#### `install(name: str, version: str, *, sandbox: Sandbox) -> InstalledScaffold`

#### `install_sidecar(*, sandbox: Sandbox) -> InstalledSidecar | None`

### `InstalledScaffold`

装好的脚手架。

- `command`：起它的命令前缀，已按 shell 转义。可执行文件是它的绝对路径；入口是 Python 模块的，是 `<python> -m <模块>`。
- `shell`：一段 `export`，放在启动命令前面，把运行时和树里的目录加进 `PATH`、`PYTHONPATH`、`LD_LIBRARY_PATH`。
  `Launch.env` 整体交给沙箱，引用不了沙箱里原来的 `$PATH`，所以走 shell。没有要设的是空串。
- `binary`：可执行文件的绝对路径；入口是 Python 模块的，是运行时的 python。`path` 是树所在的目录。
- `interpreters`：这个脚手架冻结的运行时里各个解释器的绝对路径，按运行时名字（`python`、`node`）。
  要在沙箱里起解释器就用它，不要写 `python3` / `node`：那样拿到的是题目镜像里的，
  镜像里不一定有，有也不是装这个脚手架用的那棵。

- `name: str`
- `version: str`
- `platform: str`
- `path: str`
- `binary: str`
- `command: str`
- `shell: str`
- `interpreters: Mapping[str, str]`

## 沙箱与执行

### `SandboxSpec`

环境创建参数；业务名字不决定资源或模型能力。

- `image: str`
- `cpu: int`
- `memory_gib: int`
- `storage_gib: int`
- `timeout_seconds: int`
- `network: NetworkPolicy`
- `cwd: str | None`
- `cwd_from_image: bool`
- `model: str | None`

#### `validate_workdir() -> FlowEnvironment`

### `FlowLimits`

一次 attempt 的并发实例数、累计创建数、单实例资源与流程总时限上限。

- `max_sandboxes: int`
- `max_creations: int`
- `cpu: int`
- `memory_gib: int`
- `storage_gib: int`
- `wall_seconds: int`

### `NetworkPolicy`

- `mode: NetworkMode`
- `allow: list[str]`

#### `valid_hosts(values: list[str]) -> list[str]`

#### `meaningful() -> NetworkPolicy`

### `Sandboxes`

创建和查询本次 attempt 的沙箱。名称可以重复，查询时使用 sandbox_id。

#### `create(spec: FlowEnvironment | None = None, *, name: str, **values) -> Sandbox`

按显式规格创建环境。镜像必须可在提交时冻结的依赖中解析，资源受 attempt 预算限制。

spec 可从 ctx.task.environments 读取，也可逐项提供 image/cpu/memory_gib/storage_gib/
timeout_seconds/network/cwd/cwd_from_image/model。model 为模型句柄、键或 None。

#### `acreate(spec: FlowEnvironment | None = None, *, name: str, **values) -> Sandbox`

异步创建；取消等待后 Worker 仍登记并回收迟到的创建结果。

### `Sandbox`

通过 ctx.sandboxes.create() 创建的沙箱。

- 命令用 `sh -c` 执行，工作目录默认是平台解析好的 `workdir`；非零退出不抛异常，由流程自己判断。
- `user` 是沙箱里的用户（名字或 uid），不给就是 root。
- 文件操作一律以 root 身份。
- 每条命令和它的退出码、耗时，上传下载的文件和大小，都记进 `sandbox` 日志流。

- `workdir`：平台解析好的工作目录：Task 声明的，或者镜像的 WORKDIR。
- `image_env`：镜像中配置的环境变量，执行每条命令时自动带上。
- `outbox`：这个沙箱交给流程的大文件（`Outbox`）：沙箱里的命令直传对象存储，流程读回来，不经 `download_to`。

#### `envs(env: Mapping[str, str] | None = None) -> dict[str, str]`

合并命令的环境变量；传入的同名变量覆盖镜像中的设置。

#### `run(command: str, *, timeout: float, cwd: str | None = None, env: Mapping[str, str] | None = None, user: str | int | None = None) -> CommandResult`

在 Task 的受限执行阶段里跑短命令；日志目标不决定隔离。

#### `start(command: str | list[str], *, timeout: float, **options) -> Process`

启动独立进程；同一环境的资源变更与启动原子地互斥。

#### `configure(*, network: NetworkPolicy, model: str | Model | None) -> None`

显式调整环境网络能力；有执行在运行时拒绝，避免影响其他句柄。

#### `astart(command: str | list[str], *, timeout: float, cwd: str | None = None, env: Mapping[str, str] | None = None, user: str | int | None = None, log: LogTarget | None = 'sandbox', output_limit_bytes: int = OUTPUT_LIMIT_BYTES) -> Process`

异步启动；创建期间取消也会登记并回收迟到执行。

#### `arun(command: str | list[str], *, timeout: float, **options) -> Any`

异步启动并拥有执行；调用取消时回收此执行。

#### `platform_run(command: str, *, timeout: float, cwd: str | None = None, env: Mapping[str, str] | None = None, user: str | int | None = None) -> CommandResult`

可信平台在根侧执行短命令，不应用 Task 网络策略。

仅用于装载、文件与控制操作，Task 命令用 `run`。
返回 `exit_code`、`stdout`、`stderr`；超过 `timeout` 秒抛 `TimeoutError`。

#### `run_long(command: str, *, timeout: float, cwd: str | None = None, env: Mapping[str, str] | None = None, user: str | int | None = None, log: LogTarget | None = 'sandbox', follow: str | None = None, noise: tuple[str, ...] = ()) -> LongRun`

在受限阶段里跑 Task 命令，阶段与日志目标分别指定。

#### `platform_run_long(command: str, *, timeout: float, cwd: str | None = None, env: Mapping[str, str] | None = None, user: str | int | None = None, log: LogTarget | None = 'sandbox', follow: str | None = None, noise: tuple[str, ...] = ()) -> LongRun`

可信平台在根侧运行长命令，不应用 Task 网络策略。

Task 命令用 `run_long`。每 10 秒把新增输出写进 `log` 日志流。

`follow` 是另一个要跟随的沙箱文件（命令把输出重定向到了那里，比如评估脚本的 test-stdout.txt）。
含 `noise` 里任一子串的行不进日志。超时时停掉整个进程组，返回 `timed_out=True`。

#### `upload(path: str, data: bytes | str, *, mode: int | None = None) -> None`

写一个沙箱里的文件（父目录会建出来）。`mode` 是要设的权限，比如 `0o755`。

#### `upload_file(path: str, local: Path) -> None`

把 Worker 本机的一个文件流式传进沙箱，不整份读进内存。

#### `download(path: str) -> bytes`

读一个沙箱里的文件。大文件用 `download_to`。

#### `download_to(path: str, local: Path) -> int`

把沙箱里的一个文件流式写到 Worker 本机，返回字节数。

#### `exists(path: str) -> bool`

#### `is_dir(path: str) -> bool`

#### `list(path: str, *, depth: int = 5) -> list[str]`

`path` 下的普通文件（绝对路径）；`path` 是文件时就是它自己。

#### `scratch(name: str) -> str`

沙箱里一个临时文件的绝对路径。

#### `upload_tree(local: Path, target: str, *, exclude: Sequence[str] = (), user: str | int | None = None) -> int`

把 Worker 本机的一个目录（或文件）拷进沙箱，落地后自证，返回文件数。

典型用途：把从 Agent 沙箱取出的 `/logs/artifacts` 放进 verifier 沙箱（上游 `upload_artifacts`）。

#### `download_tree(path: str, local: Path, *, exclude: Sequence[str] = ()) -> int`

把沙箱里的一个目录取到 Worker 本机的 `local` 目录下，返回文件数。

`exclude` 是 tar 的排除模式（上游 Harbor 产物声明的 `exclude`）。

#### `start_service(command: str, *, name: str, ready: str, timeout: float, env: Mapping[str, str] | None = None, user: str | int | None = None) -> Process`

启动持有独立句柄的服务；timeout 限定就绪等待，服务受流程总时限约束。

#### `end_phase() -> None`

回收环境中所有执行；下一次命令建立新的执行管理器。

#### `release() -> None`

用完了：平台收走这个沙箱的产物，然后停掉它。之后这个对象不能再用。

separate 评估前释放 Agent 沙箱（上游先停 Agent 环境再起 verifier 环境）；多步 Task 每一步的 verifier 用完就释放。
没释放的沙箱在流程结束时由平台收完产物再停。

### `Outbox`

沙箱与流程之间的大文件中转。

`upload` 把沙箱里的文件直接传上去，流程用 `open` 读回来；沙箱里要用别的工具传，用 `put_url` 签链接。

- 名字只能用字母、数字和 `. _ -`，1 到 128 个字符；同一个沙箱里同名的对象后传的覆盖先传的。
- 一个对象不超过对象存储一次 PUT 的上限（COS 是 5 GB）。
- 链接有效到这次 attempt 的墙钟截止，只能写、读或删这一个对象，签出来就进了这次 attempt 的秘密清单，事件和日志里会被替换掉。
- 对象只在这次 attempt 里用，用完 `delete`。流程被杀、没来得及删的，对象存储一天后自己删掉。

#### `put_url(name: str) -> str`

沙箱里的命令往 `name` 上传文件用的链接（HTTP PUT 整个文件，比如 `curl -fsS -T 文件 "$URL"`）。

经 `env` 交给命令，别拼进命令行。沙箱释放之后签不出来。

#### `upload(path: str, name: str) -> int`

把沙箱里的文件 `path`（相对路径按沙箱的工作目录）传成输出对象 `name`，返回字节数。

在沙箱里以 root 用 `curl` 直传对象存储，镜像里要有 curl；链接经环境变量给，不进命令行。文件超过对象存储一次 PUT 的
上限直接抛 `ValueError`，写明大小。传断了 curl 重发整个文件；最多用 30 分钟，超时抛 `TimeoutError`。

#### `open(name: str) -> io.BufferedReader`

读沙箱传上来的 `name`：只读、可 seek 的二进制文件，按范围从对象存储读，不落 Worker 本机盘。

连不上、超时、对象存储回 5xx 时从断开的位置接着读，不设次数，由流程的时限兜住；没有这个对象抛 `FileNotFoundError`。
沙箱释放之后也能读。用完关掉（`with` 语句）。

#### `delete(name: str) -> None`

删掉输出对象 `name`；没有这个对象也算删掉了。沙箱释放之后也能删。

连不上、对象存储回 5xx 或 429 时按 Worker 给的间隔退避重试，到流程时限为止；别的失败抛 `OSError`，写明状态码，不带链接。

### `Process`

运行中的进程。日志会持续收集，多次 wait() 返回同一结果，cancel() 停止进程及其子进程。

#### `on_finish(callback: Callable[[ProcessResult], None]) -> None`

登记完成通知；进程已经完成时立即调用，快速执行不会丢通知。

#### `status() -> dict[str, Any]`

读取进程状态。已结束的进程会保留最后状态，释放沙箱后仍可查询。

#### `cancel() -> dict[str, Any]`

等待这次执行与后代停止；重复调用不改变已完成结果。

#### `acancel() -> dict[str, Any]`

异步取消对应执行。

#### `wait() -> ProcessResult`

等待结果；不销毁沙箱或其他执行。

#### `await_result() -> ProcessResult`

异步观察，取消观察者不取消对应执行或共享日志读取。

#### `output(*, stdout_offset: int = 0, stderr_offset: int = 0) -> dict[str, Any]`

独立按偏移读取原始字节，不消费或重复写入公共日志。

### `ProcessResult`

进程及其子进程结束后的退出码、耗时和输出。评分由流程另行提交。

- `exit_code: int | None`
- `timed_out: bool`
- `duration: float`
- `stdout: str`
- `stderr: str`
- `output_tail: str`

### `CommandResult`

- `stdout: str`
- `stderr: str`
- `exit_code: int`

### `LongRun`

`run_long` 的结果。超时时 `exit_code` 是 None，进程已经被停掉。

- `exit_code: int | None`
- `timed_out: bool`
- `duration: float`
- `output_tail: str`

## Agent

### `Agent`

运行 Agent 命令。每次调用需要指定沙箱、模型和会话 ID。

#### `register_session(*, session_id: str, sandbox: Sandbox, model: Model, name: str, version: str, output_dir: str, converter: str) -> None`

#### `start(launch: str | Launch, *, sandbox: Sandbox, model: Model, session_id: str, timeout: float | None = None, env: Mapping[str, str] | None = None, stdin: str | None = None, cwd: str | None = None, user: str | int | None = None) -> Process`

启动 Agent 并立即返回 Process；timeout 不填时使用流程剩余时间。

#### `run(launch: str | Launch, *, sandbox: Sandbox, model: Model, session_id: str, timeout: float | None = None, **options) -> AgentRun`

启动并等待；中断调用会取消这次执行。Agent 自己以非零码退出抛 `AgentFailed`。

### `Launch`

Agent 的启动命令及运行环境。

`command` 由 bash 执行，可以包含多条命令。`stdin` 指向沙箱中的输入文件。
`cwd` 默认使用沙箱工作目录；`user` 可填用户名或 uid，默认使用 root。

- `command: str`
- `env: Mapping[str, str]`
- `stdin: str | None`
- `cwd: str | None`
- `user: str | int | None`

### `AgentRun`

Agent 的退出状态、耗时和末尾输出。

正常结束和超时都会返回此对象；非零退出会直接抛 `AgentFailed`。

- `exit_code: int | None`：正常退出是 0，超时被停掉时是 None。
- `timed_out: bool`
- `duration: float`：秒。
- `output_tail: str`：输出的最后一段（已脱敏）；完整输出在 agent 日志流里。

#### `of(result: ProcessResult) -> AgentRun`

将进程结果转成 Agent 结果；非零退出时抛 `AgentFailed`。

超时仍返回结果，让流程可以评估 Agent 已完成的工作。

#### `check() -> AgentRun`

超时抛 `AgentTimeout`，否则返回当前结果。

如果超时后还要评估已有产物，直接读取 timed_out，不调用此方法。

## 结果与主评分

### `Results`

可记录多份分析、评分或诊断。每个名字唯一，内容必须可序列化为 JSON。

结果写入本次流程的产物目录，环境释放后仍然可用。Rollout 的主评分另由 evaluation 提交。

#### `record(name: str, value: Any) -> None`

按名称保存一份 JSON 结果。同一次流程中不能重复使用名称；主评分通过 evaluation 提交。

### `Evaluation`

`ctx.evaluation`：这次 Rollout 的评估结论，**只能报一次**。

平台怎么结算：

- `score(rewards)`：平台查每个值都是有限数、有 `reward` 键、`reward` 不是负数，过了就是
  `resolved`（≥ 1）或 `unresolved`；没过是 `no_reward`，类别是 `reward_non_finite` /
  `reward_missing_key` / `reward_negative`。
- `fail("verifier_timeout", message)`：测试超过规定时限，平台计 0 分、`unresolved`。
- `fail(category, message)` 的其余类别：`no_reward`，结果码 `EVALUATION_FAILED`。
- `skip()`：Task 有评估声明时是 `no_reward`（`flow_skipped`）；没有评估声明时不影响结果。
- 什么都不报：Task 有评估声明时流程正常结束是 `flow_no_verdict`，出错是 `flow_failed`。

- `given: bool`：已经报过结论了没有。

#### `score(rewards: Mapping[str, float], *, test_exit_code: int | None = None, detail: str = '') -> None`

评出了分数。`rewards` 原样保存，主分是 `rewards["reward"]`。

#### `fail(category: str, message: str = '', *, test_exit_code: int | None = None, detail: str = '') -> None`

上报评估未完成的原因。`category` 取上游 Harbor 的类别之一：

`tests_upload_failed`、`verifier_env_failed`、`verifier_timeout`、`reward_missing`、
`reward_empty`、`reward_unparsable`、`reward_non_finite`、`reward_missing_key`、`reward_negative`。
`verifier_timeout` 专指测试进程耗尽测试时限，平台按未通过计 0 分；沙箱启动、连接和流程超时不能用它。
`detail` 是原始内容（解析不了的 reward 文件、错误输出），会脱敏、截断后进事件。

#### `skip(reason: str = '') -> None`

这次不评估。

#### `submit(verdict: Mapping[str, Any]) -> None`

用一个字典报结论：`{"status": "scored", "rewards": {...}}`、
`{"status": "failed", "category": ..., "message": ...}` 或 `{"status": "skipped"}`。

## 文件和产物

读取任务和流程文件，校验内容摘要，并将文件复制到沙箱。

### `Files`

一个流程可读取和投递的目录。

#### `sub(path: str) -> Files`

以 `path` 为根的同一种只读目录。

#### `copy_to_sandbox(path: str, target: str, *, sandbox: Sandbox, readonly: bool = False, exclude: Sequence[str] = (), user: str | int | None = None, verify: bool = True) -> Copied`

把一个文件或一棵目录拷进沙箱，目标由 sandbox 指定。

- 目录的**内容**落在 `target` 下（`target` 里原有的文件保留）；文件拷成 `target` 这个文件。
- `exclude` 是 fnmatch 模式，匹配相对路径或其中任一段。
- `user` 给了就把拷进去的东西交给这个用户；`readonly` 为真时收掉写权限。
- `verify` 为真时在沙箱里把刚落地的文件重新打包取回，按素材的算法重算树摘要，对不上就抛异常。
- 在时间线上记一条 `files_copied`：来源、目标、树摘要、条目数。

#### `path(path: str = '.') -> Path`

这个条目在 Worker 本机上的路径（只读）。

#### `exists(path: str) -> bool`

#### `is_dir(path: str) -> bool`

#### `list(path: str = '.') -> builtins.list[str]`

一层目录里的条目，目录名以 `/` 结尾。

#### `walk(path: str = '.') -> builtins.list[str]`

`path` 下所有文件的相对路径（相对这个目录的根），排好序。

#### `glob(pattern: str) -> builtins.list[str]`

按 `fnmatch` 模式匹配 `walk` 的结果，比如 `"skills/*/SKILL.md"`。

#### `stat(path: str) -> FileInfo`

#### `read(path: str) -> bytes`

#### `read_text(path: str, encoding: str = 'utf-8') -> str`

#### `open(path: str) -> IO[bytes]`

以二进制方式打开一个文件；用完要关。

#### `materialize(path: str, destination: Path) -> Path`

把一个文件或一棵目录拷到 Worker 本机的 `destination`（通常在 `ctx.tmp` 下），返回拷出来的位置。

### `TaskFiles`

按冻结清单读取、核对和投递 Task 文件。

#### `sub(path: str) -> TaskFiles`

#### `digest(path: str = '.') -> str`

`path`（目录或文件）的内容摘要：按冻结清单里的相对路径、可执行位、内容摘要和软链接算，不读文件。
任何一个文件、可执行位或软链接变了，摘要就变；和 `copy_to_sandbox` 返回的 `sha256` 相同。

#### `copy_to_sandbox(path: str, target: str, *, sandbox: Sandbox, readonly: bool = False, exclude: Sequence[str] = (), user: str | int | None = None, verify: bool = True) -> Copied`

和 `copy_to_sandbox` 一样，但不经 envd 传：Worker 按清单打包、核对内容，放进这个沙箱的投递目录，
沙箱里解包后数文件、软链接、可执行文件。对不上抛 `TaskFileMismatch`。`verify` 不起作用，
数量总是要核对。

#### `path(path: str = '.') -> Path`

本机路径。文件会先核对一次；之后直接读这个路径不再核对，要核对就用 `read`。

#### `exists(path: str) -> bool`

#### `is_dir(path: str) -> bool`

#### `list(path: str = '.') -> builtins.list[str]`

#### `walk(path: str = '.') -> builtins.list[str]`

#### `stat(path: str) -> FileInfo`

#### `read(path: str) -> bytes`

#### `open(path: str) -> IO[bytes]`

核对过的内容，以二进制文件对象给出。

### `FileInfo`

一个条目的样子。

- `path: str`
- `size: int`
- `is_dir: bool`
- `executable: bool`

### `Copied`

`copy_to_sandbox` 的结果：落在哪、树摘要（和素材同一个算法）、几个条目。

- `target: str`
- `sha256: str`
- `file_count: int`

### `Materials`

提交时冻结的素材。下发目标环境由流程显式选择。

#### `copy_to(sandbox: Sandbox) -> dict[str, Any]`

把本次冻结的素材按其目标路径和权限下发到指定环境。

Worker 把打包件链接进这个沙箱的投递目录，这里按声明顺序解包、核对数量，解完不管成败都让 Worker 删掉链接：
Agent 起来时投递目录里只剩标记文件。解出来的数量和冻结的对不上抛 `TaskFileMismatch`。

### `Artifacts`

用 collect() 登记要保存的产物，用 capture() 立即保存文件快照。

#### `collect(path: str, *, sandbox: Sandbox) -> None`

登记沙箱中的文件或目录，在沙箱释放前收集到本次产物中。

#### `capture(sandbox: Sandbox, *, path: str) -> ArtifactRef`

立即保存文件快照。返回后可释放源沙箱，再将快照复制到其他沙箱。

### `ArtifactRef`

已保存的文件快照，包括内容摘要、大小和文件数。

- `artifact_id: str`
- `sha256: str`
- `size_bytes: int`
- `file_count: int`
- `kind: str`

#### `materialize(sandbox: Sandbox, *, target: str) -> dict[str, Any]`

将快照复制到目标沙箱。之后修改目标文件不会改变快照。

## Aster 接口

### `AsterApi`

`ctx.api`：用这次 attempt 的令牌调 Aster 接口，令牌代表 Run 的提交人，范围是 `runs:read`、`runs:write`，
attempt 结束时撤销。

连不上、Server 回 502/503/504、错误里写着可以重发时，按 Worker 给的间隔退避重试，到这次 attempt 的墙钟截止为止（SG-14）；
别的错误直接抛 `AsterApiError`。

#### `get(path: str, **query) -> Any`

`GET path`，返回回复里的 `data`。`path` 从 `/api/v1/` 开始，比如
`ctx.api.get(f"/api/v1/datasets/versions/{version_id}", task_key=ctx.task.key)`。

### `AsterApiError`

Aster 接口调用失败。code 为错误码，retryable 表示是否可重试；连接失败的 code 是 UNREACHABLE。

## 异常

### `TempFail`（异常）

临时故障，比如注册表超时、被限流。

第一次 `ctx.agent.run` 之前抛它，结果码是 `FLOW_SETUP_TEMPFAIL`；其他异常是 `FLOW_SETUP_FAILED`。
两种都在 Agent 启动之前，平台都会在同一轮里重来；结果码只告诉看的人是不是临时故障。
Agent 启动之后抛它和抛别的异常一样。

### `ModelRejected`（异常）

模型配置被模型端拒了（key 错、模型名错、地址错，4xx 里除了 429），或开了 TITO 而模型后端不是能从 token id 生成的 sglang。
`Model.check()`、`Model.connect()` 抛它；重来还是一样，第一次起 Agent 之前抛出时结果码是 `FLOW_SETUP_FAILED`。

### `AgentTimeout`（异常）

Agent 超过了时限。`AgentRun.check()` 抛它；流程不接住，而这次没有冻结评估时，结果码是 `AGENT_TIMEOUT`。

### `AgentFailed`（异常）

Agent 自己以非零码退出。`ctx.agent.run` 和 Agent 会话的 `run`、`wait` 抛它，流程就此停下。

结果码是 `AGENT_FAILED`，不评估；流程接住它也改不了结果，平台按记下的退出码判。

### `TaskFileMismatch`（异常）

读取或打包的文件与冻结清单不一致。

### `SandboxNotReady`（异常）

沙箱起来了，但还不能交给流程用：投递目录挂的不对、素材下发不成、工作目录读不出来。

结果码是 `SANDBOX_PREPARE_FAILED`。内容对不上是另一回事，那是 `DeliveryMismatch`。

### `TaskConfigNotDeclared`（异常）

读了流程没在 `@flow(requires=...)` 里声明的 Task 配置。提交时没检查也没冻结它，平台给不出来。

### `VerifierUnavailable`（异常）

平台没能起出 verifier 沙箱（云报错、隔网自证不过）。`harbor` 预置流程把它报成 `verifier_env_failed`。

### `SandboxReleased`（异常）

这个沙箱已经 `release()` 过，不能再用。
