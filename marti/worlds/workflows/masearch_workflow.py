"""Workflow: Multi-agent Search (MASearch)
Pattern: Prompt Engineer -> Planner -> Retriever (iterative JSON tool calls) -> Generator
"""
import os
from typing import Dict, List, Any, Optional
import asyncio
from marti.helpers.logging import init_logger
from marti.verifiers.auto_verify import auto_verify
from marti.worlds.workflows.utils import apply_template_with_tokenizer
from marti.worlds.steps.mcp_step import step_with_tools

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))

# --- Main Workflow -----------------------------------------------------------
async def workflow(
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    tool_manager,
    task: str,
    metadata: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """MASearch workflow orchestrating 4 agents.

    Expected agents list order:
      0: Prompt Engineer
      1: Planner
      2: Retriever
      3: Generator
    """
    assert tool_manager is not None, "tool_manager required"
    assert len(agents) >= 4, "Need four agents for MASearch"

    prompt_engineer, planner, retriever, generator = agents[:4]
    trajectory: List[Dict[str, Any]] = []

    # ---- Prompt Engineer ----
    pe_input = apply_template_with_tokenizer(
        prompt_engineer["tokenizer"],
        prompt_engineer["chat_template"].format(question=prompt)
    )
    pe_resp = await prompt_engineer["llm"].generate_async.remote(
        pe_input,
        prompt_engineer["sampling_params"]
    )
    pe_text = pe_resp.outputs[0].text.strip()
    trajectory.append({
        "turn_id": 0,
        "agent_index": 0,
        "agent_name": prompt_engineer["agent_id"],
        "agent_role": prompt_engineer["agent_role"],
        "agent_input": pe_input,
        "agent_output": pe_text,
        "metadata": {"original_prompt": prompt}
    })

    # ---- Planner ----
    planner_input = apply_template_with_tokenizer(
        planner["tokenizer"],
        planner["chat_template"].format(prompt=pe_text)
    )
    planner_resp = await planner["llm"].generate_async.remote(
        planner_input,
        planner["sampling_params"]
    )
    planner_text = planner_resp.outputs[0].text
    trajectory.append({
        "turn_id": 1,
        "agent_index": 1,
        "agent_name": planner["agent_id"],
        "agent_role": planner["agent_role"],
        "agent_input": planner_input,
        "agent_output": planner_text,
        "metadata": {"original_prompt": prompt, "normalized_prompt": pe_text}
    })

    # ---- Retriever Loop (JSON tool calls parsed by ToolParser via step_with_tools) ----
    # Include original and normalized prompt context for retriever
    retrieval_observation: List[str] = [
        f"<original_prompt>\n{prompt}\n</original_prompt>",
        f"<normalized_prompt>\n{pe_text}\n</normalized_prompt>",
        planner_text
    ]
    max_retrieval_turns = tool_manager.get_max_turns()
    ready = False

    for r_turn in range(max_retrieval_turns):
        retriever_prompt_text = retriever["chat_template"].format(plan='\n'.join(retrieval_observation))
        retriever_input = apply_template_with_tokenizer(
            retriever["tokenizer"], retriever_prompt_text
        )
        retriever_resp = await retriever["llm"].generate_async.remote(
            retriever_input,
            retriever["sampling_params"]
        )
        retriever_output = retriever_resp.outputs[0].text

        if "<ready/>" in retriever_output:
            ready = True
            trajectory.append({
                "turn_id": len(trajectory),
                "agent_index": 2,
                "agent_name": retriever["agent_id"],
                "agent_role": retriever["agent_role"],
                "agent_input": retriever_input,
                "agent_output": retriever_output,
                "metadata": {"retrieval_turn": r_turn, "ready": True, "original_prompt": prompt}
            })
            break

        step_result = await step_with_tools(
            observation=retrieval_observation,
            action=retriever_output,
            tool_manager=tool_manager,
            metadata=metadata or {}
        )
        retrieval_observation = step_result["next_observation"]

        trajectory.append({
            "turn_id": len(trajectory),
            "agent_index": 2,
            "agent_name": retriever["agent_id"],
            "agent_role": retriever["agent_role"],
            "agent_input": retriever_input,
            "agent_output": retriever_output,
            "metadata": {"retrieval_turn": r_turn, **step_result.get("extra_logs", {}), "original_prompt": prompt}
        })

        if step_result.get("done"):
            ready = True
            break

    # ---- Generator ----
    generator_context = "\n".join(retrieval_observation)
    generator_input_base = generator["chat_template"].format(question=prompt)
    generator_input = apply_template_with_tokenizer(
        generator["tokenizer"], generator_input_base
    )
    generator_input_full = generator_input + f"\n<context>\n{generator_context[:4000]}\n</context>"
    gen_resp = await generator["llm"].generate_async.remote(
        generator_input_full,
        generator["sampling_params"]
    )
    gen_text = gen_resp.outputs[0].text
    trajectory.append({
        "turn_id": len(trajectory),
        "agent_index": 3,
        "agent_name": generator["agent_id"],
        "agent_role": generator["agent_role"],
        "agent_input": generator_input_full,
        "agent_output": gen_text,
        "metadata": {"retrieval_ready": ready, "original_prompt": prompt, "normalized_prompt": pe_text}
    })

    # ---- Verification (score subset: prompt engineer, planner, generator) ----
    outputs_for_scoring = [t for t in trajectory if t["agent_index"] in (0, 1, 3)]
    try:
        scores = auto_verify(
            task,
            1,
            [t["agent_output"] for t in outputs_for_scoring],
            [label] * len(outputs_for_scoring)
        )
    except Exception:
        scores = [0.0] * len(outputs_for_scoring)

    si = 0
    for t in trajectory:
        if t["agent_index"] in (0, 1, 3):
            t["agent_reward"] = scores[si]
            si += 1

    return {
        "prompt": prompt,
        "normalized_prompt": pe_text,
        "label": label,
        "trajectory": trajectory,
        "final_reward": scores[-1] if scores else 0.0
    }