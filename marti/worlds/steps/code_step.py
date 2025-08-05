import json
from typing import Dict, Any, List, Optional

from marti.worlds.tools.manager import ToolManager
from marti.helpers.logging import init_logger
from marti.verifiers.deepcoder.code_reward import rllm_reward_fn_code

import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))


async def step(
    observation: List[str],
    action: str,
    tool_manager: Optional[ToolManager] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generic step function that works with any tools.
    Tool parser should return: [{"name": "tool_name", "args": "{...}"}, ...]
    """
    metadata = json.loads(kwargs["metadata"])
    final_reward = rllm_reward_fn_code(
        metadata["data_source"], action, kwargs["label"])

    return {
        "next_observation": observation + [action],
        "done": True,
        "extra_logs": {"tools_used": {}},
        "final_reward": final_reward
    }
