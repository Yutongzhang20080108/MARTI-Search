import json
from typing import Dict, Any, List, Optional

from marti.worlds.tools.manager import ToolManager
from marti.helpers.logging import init_logger

import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))

async def step_with_tools(
    observation: List[str],
    action: str,
    tool_manager: Optional[ToolManager] = None,
    **kwargs):
    """
    """
    pass