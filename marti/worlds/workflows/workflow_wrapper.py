import time
import ray
import asyncio
import copy
from typing import Dict, List, Any, Optional
from marti.helpers.logging import init_logger
from marti.verifiers.auto_verify import auto_verify

logger = init_logger(__name__)

@ray.remote
class WorkflowInstance:
    """Workflow instance that orchestrates multi-agent interactions."""
    
    def __init__(self, workflow_func_path: str, agents: List[Dict[str, Any]]):
        """
        Initialize workflow instance with a workflow function and agent configurations.
        
        Args:
            workflow_func_path: Path to the workflow function file
            agents: List of agent configurations with their LLMRayActorAsync instances
        """
        self.agents = agents

        if workflow_func_path.endswith(".py"):
            import importlib.util
            spec = importlib.util.spec_from_file_location("workflow", workflow_func_path)
            workflow_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(workflow_module)
            self.workflow_func = workflow_module.workflow
        else:
            raise ValueError("Workflow path must be a Python file")

    async def execute(self, prompt: str, label: str, tool_manager, **kwargs) -> Dict[str, Any]:
        """Execute the workflow with the given prompt."""
        return await self.workflow_func(
            prompt=prompt,
            label=label,
            agents=self.agents,
            tool_manager=tool_manager,
            **kwargs
        )

@ray.remote
class MultiAgentWrapper:
    """Wrapper for managing multiple agents and their interactions."""
    
    def __init__(self, agents: List[Dict[str, Any]], workflow_args, *args, **kwargs):
        """
        Initialize multi-agent wrapper.
        
        Args:
            agents: List of agent configurations, each containing:
                - name: Agent identifier
                - llm: LLMRayActorAsync instance
                - tokenizer: Tokenizer for the agent
                - sampling_params: Default sampling parameters
                - workflow_func_path: Path to agent step function (optional)
        """
        self.agents = agents
        self.workflow_args = workflow_args
        self.workflow_func_path = kwargs.pop("workflow_func_path")
        self.result_queue = asyncio.Queue()

    async def add_requests(
        self,
        tool_manager,
        prompts: List[str],
        labels: List[str],
        task: str,
        metadata: Optional[List[Dict]] = None
    ):
        """Process requests using multi-agent workflow."""
        
        # Create semaphore to control concurrent workflow execution
        semaphore = asyncio.Semaphore(tool_manager.get_num_workers())
        
        async def execute_workflow(prompt: str, label: str, meta: Optional[Dict] = None):
            """Execute a single workflow instance."""
            async with semaphore:
                workflow_start = time.time()
                
                # Create workflow instance
                workflow_instance = WorkflowInstance.remote(
                    self.workflow_func_path,
                    self.agents
                )

                try:
                    # Execute workflow
                    result = await workflow_instance.execute.remote(
                        prompt=prompt,
                        label=label,
                        tool_manager=tool_manager,
                        task=task,
                        metadata=meta,
                        workflow_args=self.workflow_args
                    )
                    
                    # Add timing information
                    result["workflow_time"] = time.time() - workflow_start

                    # Store result
                    await self.result_queue.put(result)

                except Exception as e:
                    logger.error(f"Workflow execution error: {e}")
                    error_result = {
                        "prompt": prompt,
                        "label": label,
                        "error": str(e),
                        "workflow_time": time.time() - workflow_start
                    }
                    await self.result_queue.put(error_result)

                finally:
                    ray.kill(workflow_instance)

        # Create tasks for all workflows
        if metadata is None:
            metadata = [{} for _ in range(len(prompts))]
            
        tasks = []
        for prompt, label, meta in zip(prompts, labels, metadata):
            tasks.append(execute_workflow(prompt, label, copy.deepcopy(meta)))
        
        # Execute all workflows concurrently
        await asyncio.gather(*tasks)
    
    # async def get_responses(self) -> List[Dict[str, Any]]:
    #     """Get all completed workflow results."""
        # results = []
        # while not self.result_queue.empty():
        #     try:
        #         results.append(await self.result_queue.get())
        #     except asyncio.QueueEmpty:
        #         break
        # return results

    async def get_responses(self, expected_len: int) -> List[Dict[str, Any]]:
        results = []
        for _ in range(expected_len):
            results.append(await self.result_queue.get())
        return results