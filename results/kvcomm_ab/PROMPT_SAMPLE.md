# 真实多智能体推理 Prompt 样例
> 全部真实数据：SWE-Smith pandas manifest + 真实 pandas 源码 + 驱动器精确逻辑（`--position-shift --partial-share`）重建输入；输出取自 `results/kvcomm_ab/7b_lossless/outputs.jsonl`（Qwen2.5-Coder-7B 真实推理产物）。 未起服务器。
- **case_id**: `pandas-dev__pandas.95280573.combine_file__11s6papj`
- **segment_count**: 5（每个 code_base slot = 一个真实 pandas 文件，截断 8000 字符）
- **agent_count**: 5（角色链 implementer → debugger → reviewer → verifier → auditor）
- **per-agent 变换**: `--position-shift` 旋转 slot 顺序 + `--partial-share` 丢 1 个顶层函数
- **复用机制**: agent 1=source；agent 2..5 用不同 `cache_salt` → radix 冷 → 只能靠 KVCOMM 跨位置 copy+RoPE 复用
- **代码 slot 在本样例中截断到前 12 行**（真实 benchmark 不截断）

---

## Agent 1 — implementer

- slot 顺序（旋转后）: `column.py → buffer.py → dataframe.py → dataframe_protocol.py → from_dataframe.py`
- partial-share 丢弃函数: `from_dataframe`

### Input

**system**:
```
You are a senior software engineering agent.
```

**user**:
```
## Agent role
implementer
## Case
pandas-dev__pandas.95280573.combine_file__11s6papj
## Instruction
Inspect the repeated repository code and answer with one concise implementation risk.
## code_base1: pandas/core/interchange/column.py
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas._libs.lib import infer_dtype
from pandas._libs.tslibs import iNaT
from pandas.errors import NoBufferPresent
…(60 more lines truncated)
## code_base2: pandas/core/interchange/buffer.py
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas.core.interchange.dataframe_protocol import (
    Buffer,
    DlpackDeviceType,
)

…(64 more lines truncated)
## code_base3: pandas/core/interchange/dataframe.py
from __future__ import annotations

from collections import abc
from typing import TYPE_CHECKING

from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
from pandas.core.interchange.utils import maybe_rechunk

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
…(11 more lines truncated)
## code_base4: pandas/core/interchange/dataframe_protocol.py
"""
A verbatim copy (vendored) of the spec from https://github.com/data-apis/dataframe-api
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import enum
from typing import (
…(239 more lines truncated)
## code_base5: pandas/core/interchange/from_dataframe.py
from __future__ import annotations

import ctypes
import re
from typing import (
    Any,
    overload,
)

import numpy as np

from pandas._config import using_string_dtype
…(175 more lines truncated)
## Upstream context
Planner cached exact repository code objects for downstream agents. Previous agent index: 0.
## Output
Return exactly one short sentence for agent 1.
```

### Output

（真实 7B 输出）:
```
Memory management risk due to potential buffer leaks if not properly handled.
```

---

## Agent 2 — debugger

- slot 顺序（旋转后）: `buffer.py → dataframe.py → dataframe_protocol.py → from_dataframe.py → column.py`
- partial-share 丢弃函数: `__init__`

### Input

**system**:
```
You are a senior software engineering agent.
```

**user**:
```
## Agent role
debugger
## Case
pandas-dev__pandas.95280573.combine_file__11s6papj
## Instruction
Inspect the repeated repository code and answer with one concise implementation risk.
## code_base1: pandas/core/interchange/buffer.py
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas.core.interchange.dataframe_protocol import (
    Buffer,
    DlpackDeviceType,
)

…(108 more lines truncated)
## code_base2: pandas/core/interchange/dataframe.py
from __future__ import annotations

from collections import abc
from typing import TYPE_CHECKING

from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
from pandas.core.interchange.utils import maybe_rechunk

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
…(92 more lines truncated)
## code_base3: pandas/core/interchange/dataframe_protocol.py
"""
A verbatim copy (vendored) of the spec from https://github.com/data-apis/dataframe-api
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import enum
from typing import (
…(239 more lines truncated)
## code_base4: pandas/core/interchange/from_dataframe.py
from __future__ import annotations

import ctypes
import re
from typing import (
    Any,
    overload,
)

import numpy as np

from pandas._config import using_string_dtype
…(214 more lines truncated)
## code_base5: pandas/core/interchange/column.py
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas._libs.lib import infer_dtype
from pandas._libs.tslibs import iNaT
from pandas.errors import NoBufferPresent
…(190 more lines truncated)
## Upstream context
Planner cached exact repository code objects for downstream agents. Previous agent index: 1.
## Output
Return exactly one short sentence for agent 2.
```

### Output

（真实 7B 输出）:
```
Zero-copy approach is requested but not supported for non-contiguous buffers.
```

---

## Agent 3 — reviewer

- slot 顺序（旋转后）: `dataframe.py → dataframe_protocol.py → from_dataframe.py → column.py → buffer.py`
- partial-share 丢弃函数: `bufsize`

### Input

**system**:
```
You are a senior software engineering agent.
```

**user**:
```
## Agent role
reviewer
## Case
pandas-dev__pandas.95280573.combine_file__11s6papj
## Instruction
Inspect the repeated repository code and answer with one concise implementation risk.
## code_base1: pandas/core/interchange/dataframe.py
from __future__ import annotations

from collections import abc
from typing import TYPE_CHECKING

from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
from pandas.core.interchange.utils import maybe_rechunk

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
…(97 more lines truncated)
## code_base2: pandas/core/interchange/dataframe_protocol.py
"""
A verbatim copy (vendored) of the spec from https://github.com/data-apis/dataframe-api
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import enum
from typing import (
…(239 more lines truncated)
## code_base3: pandas/core/interchange/from_dataframe.py
from __future__ import annotations

import ctypes
import re
from typing import (
    Any,
    overload,
)

import numpy as np

from pandas._config import using_string_dtype
…(209 more lines truncated)
## code_base4: pandas/core/interchange/column.py
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas._libs.lib import infer_dtype
from pandas._libs.tslibs import iNaT
from pandas.errors import NoBufferPresent
…(204 more lines truncated)
## code_base5: pandas/core/interchange/buffer.py
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas.core.interchange.dataframe_protocol import (
    Buffer,
    DlpackDeviceType,
)

…(120 more lines truncated)
## Upstream context
Planner cached exact repository code objects for downstream agents. Previous agent index: 2.
## Output
Return exactly one short sentence for agent 3.
```

### Output

（真实 7B 输出）:
```
The codebase contains multiple implementations of the same functionality, which can lead to inconsistencies and maintenance issues.
```

---

## Agent 4 — verifier

- slot 顺序（旋转后）: `dataframe_protocol.py → from_dataframe.py → column.py → buffer.py → dataframe.py`
- partial-share 丢弃函数: `metadata`

### Input

**system**:
```
You are a senior software engineering agent.
```

**user**:
```
## Agent role
verifier
## Case
pandas-dev__pandas.95280573.combine_file__11s6papj
## Instruction
Inspect the repeated repository code and answer with one concise implementation risk.
## code_base1: pandas/core/interchange/dataframe_protocol.py
"""
A verbatim copy (vendored) of the spec from https://github.com/data-apis/dataframe-api
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import enum
from typing import (
…(239 more lines truncated)
## code_base2: pandas/core/interchange/from_dataframe.py
from __future__ import annotations

import ctypes
import re
from typing import (
    Any,
    overload,
)

import numpy as np

from pandas._config import using_string_dtype
…(226 more lines truncated)
## code_base3: pandas/core/interchange/column.py
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas._libs.lib import infer_dtype
from pandas._libs.tslibs import iNaT
from pandas.errors import NoBufferPresent
…(203 more lines truncated)
## code_base4: pandas/core/interchange/buffer.py
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas.core.interchange.dataframe_protocol import (
    Buffer,
    DlpackDeviceType,
)

…(120 more lines truncated)
## code_base5: pandas/core/interchange/dataframe.py
from __future__ import annotations

from collections import abc
from typing import TYPE_CHECKING

from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
from pandas.core.interchange.utils import maybe_rechunk

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
…(99 more lines truncated)
## Upstream context
Planner cached exact repository code objects for downstream agents. Previous agent index: 3.
## Output
Return exactly one short sentence for agent 4.
```

### Output

（真实 7B 输出）:
```
The codebase contains multiple implementations of similar functionality, which can lead to inconsistencies and maintenance issues.
```

---

## Agent 5 — auditor

- slot 顺序（旋转后）: `from_dataframe.py → column.py → buffer.py → dataframe.py → dataframe_protocol.py`
- partial-share 丢弃函数: `num_columns`

### Input

**system**:
```
You are a senior software engineering agent.
```

**user**:
```
## Agent role
auditor
## Case
pandas-dev__pandas.95280573.combine_file__11s6papj
## Instruction
Inspect the repeated repository code and answer with one concise implementation risk.
## code_base1: pandas/core/interchange/from_dataframe.py
from __future__ import annotations

import ctypes
import re
from typing import (
    Any,
    overload,
)

import numpy as np

from pandas._config import using_string_dtype
…(216 more lines truncated)
## code_base2: pandas/core/interchange/column.py
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas._libs.lib import infer_dtype
from pandas._libs.tslibs import iNaT
from pandas.errors import NoBufferPresent
…(181 more lines truncated)
## code_base3: pandas/core/interchange/buffer.py
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas.core.interchange.dataframe_protocol import (
    Buffer,
    DlpackDeviceType,
)

…(120 more lines truncated)
## code_base4: pandas/core/interchange/dataframe.py
from __future__ import annotations

from collections import abc
from typing import TYPE_CHECKING

from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
from pandas.core.interchange.utils import maybe_rechunk

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
…(101 more lines truncated)
## code_base5: pandas/core/interchange/dataframe_protocol.py
"""
A verbatim copy (vendored) of the spec from https://github.com/data-apis/dataframe-api
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import enum
from typing import (
…(239 more lines truncated)
## Upstream context
Planner cached exact repository code objects for downstream agents. Previous agent index: 4.
## Output
Return exactly one short sentence for agent 5.
```

### Output

（真实 7B 输出）:
```
The codebase contains multiple implementations of the same functionality, which can lead to inconsistencies and maintenance issues.
```

---

