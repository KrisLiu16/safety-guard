# aster_flow.definition

单文件流程的入口声明。

## 源码

```python
"""单文件流程的入口声明。"""

from collections.abc import Callable, Sequence
from typing import TypeVar

from aster_runtime.flow import FlowLimits, task_requirements

F = TypeVar("F", bound=Callable[..., object])


def flow(
    *,
    setup_timeout: int,
    finish_timeout: int,
    atif: str = "none",
    images: dict[str, str] | None = None,
    limits: dict[str, int] | None = None,
    requires: Sequence[str] = (),
) -> Callable[[F], F]:
    """声明 run(ctx) 的准备/收尾时限（秒）、轨迹转换方式和用到的 Task 配置。

    两个时限必须显式填写，范围 1–86400。使用单个标准 Agent 时，平台从适配器
    推导轨迹转换；自行生成轨迹可指定 atif="file"。声明由服务端静态读取，不执行源码。

    ``requires`` 列出流程读的 Task 配置，取值 ``instruction``、``agent_environment``、``evaluation``。
    提交时只检查和冻结声明了的：没声明的 Task 配置不会拦下提交，``ctx.task`` 上也读不到。
    声明 ``evaluation`` 必须同时声明 ``agent_environment``；没声明 ``agent_environment`` 时必须写 ``limits``。
    """
    for value in (setup_timeout, finish_timeout):
        if type(value) is not int or not 1 <= value <= 86400:
            raise ValueError("流程时限必须是 1–86400 的整数秒")

    if limits is not None:
        FlowLimits.model_validate(limits)
    task_requirements(requires, limits=limits)

    def decorate(fn: F) -> F:
        return fn

    return decorate
```
