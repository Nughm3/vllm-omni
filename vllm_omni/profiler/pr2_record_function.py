# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PR #2677 perf experiment: always emit torch profiler spans.

vLLM's record_function_or_nullcontext is a no-op unless
VLLM_CUSTOM_SCOPES_FOR_PROFILING=1. Stage worker processes may not
inherit that env reliably in our bench harness, so PR2-instrumented code
imports this helper instead.
"""

from __future__ import annotations

from contextlib import AbstractContextManager

from torch.profiler import record_function


def record_function_or_nullcontext(name: str) -> AbstractContextManager:
    return record_function(name)
