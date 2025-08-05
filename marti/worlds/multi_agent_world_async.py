# marti/marti/models/vllm/multi_agent_engine_async.py
import torch
import asyncio
import os
import time
import ray
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum
import copy
import importlib
import torch
from vllm import SamplingParams

from marti.models.model_utils import process_sequences
from marti.worlds.base_world import Samples

from marti.helpers.logging import init_logger
from marti.models.vllm.engine_async import LLMRayActorAsync, AgentInstance, get_tokenize_text_len
from marti.verifiers.auto_verify import auto_verify
from marti.worlds.base_world import BaseWorld

from marti.worlds.workflows.workflow_wrapper import MultiAgentWrapper
from marti.worlds.workflows.default_processor import processor
from marti.worlds.tools.manager import ToolManager
from marti.worlds.tools.mcp_manager import MCPManager
from marti.worlds.tool_world import register_mcp_tools, register_openai_tools, print_tools

class MultiAgentWorldAsync(BaseWorld):
    def __init__(self, strategy, agents, *args, **kwargs):
        super().__init__(strategy, agents, *args, **kwargs)
        """
        agents: List[Dict[str, Any]]
             {
                "agent_id": unique agent id
                "agent_role": agent role (generator/refiner/verifier/coder/...)
                "pretrain": path to pretrain models
                "llms": a list of vllm engines
                "tokenizer": hf tokenizer
                "generate_kwargs": generate kwargs, which is different from vllm.SamplingParams
                "is_reasoning_model": reasoning model with <think> tags or not
            }
        """

        self.workflow_args = self.args.get("workflow_args", {})
        print("workflow args", self.workflow_args)
        self.num_agents = len(self.agents)
    
        self._init_tool_manager()
        self._init_processor()

    def _init_tool_manager(self):
        self.tools_config = self.args.get("tools_config", {})

        assert self.packing_samples, "Only support packing samples"

        if self.tools_config.get("mcp_url", None) is not None:
            self.tool_manager = MCPManager(self.tools_config)
            self.tools = register_mcp_tools(self.tool_manager)
        else:
            self.tool_manager = ToolManager(self.tools_config)
            self.tools = register_openai_tools(self.tools_config, self.tool_manager)

        print_tools(self.tools)

    def _init_processor(self):
        """
        We convert collected workflow trajectories into training samples for each agent with the given processor
        """
        if self.args.processor_func_path is None:
            self.processor = processor
        if self.args.processor_func_path.endswith(".py"):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "processor", self.args.processor_func_path)
            processor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(processor_module)
            self.processor = processor_module.processor
        else:
            raise ValueError("Processor path must be a Python file")

    def get_rank_agent(self, rank, world_size, is_eval=False):
        """
        Get the first llm for async request
        """
        rank_agents = [{} for _ in range(self.num_agents)]
        for aid, agent in enumerate(self.agents):
            agent_llms = agent["llms"]
            if len(agent_llms) <= world_size:
                llms = [agent_llms[rank % len(agent_llms)]]
            else:
                llms = agent_llms[rank::world_size]

            generate_kwargs = agent["generate_kwargs"]
            sampling_params = SamplingParams(
                temperature=generate_kwargs.get(
                    "eval_temperature" if is_eval else "temperature", 1.0),
                top_p=generate_kwargs.get("top_p", 1.0),
                top_k=generate_kwargs.get("top_k", -1),
                max_tokens=generate_kwargs.get("max_new_tokens", 1024),
                min_tokens=generate_kwargs.get("min_new_tokens", 16),
                skip_special_tokens=generate_kwargs.get(
                    "skip_special_tokens", False),
                include_stop_str_in_output=True,
                truncate_prompt_tokens=self.args.prompt_max_len if self.args.truncate_prompt else None)

            agent_dict = {
                "llm": llms[0],
                "sampling_params": sampling_params
            }
            for use_key in ["agent_id", "agent_role", "tokenizer", "is_reasoning_model"]:
                agent_dict[use_key] = deepcopy(agent[use_key])

            rank_agents[aid] = agent_dict
        return rank_agents

    def tokenize_fn_with_tok(self, messages, tokenizer=None, max_len=None):
        tokenizer = self.shared_tokenizer if tokenizer is None else tokenizer
        # For inputs
        if isinstance(messages, list):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompt_max_len = self.args.prompt_max_len if max_len is None else max_len
        # For outputs
        elif isinstance(messages, str):
            prompt = messages
            prompt_max_len = self.args.generate_max_len if max_len is None else max_len
        else:
            raise NotImplementedError

        return self.tokenize_fn(tokenizer, prompt, prompt_max_len, padding=False)["input_ids"]

    def distribute_prompts(self, task, prompts, labels, metadata, rank_agents_list, is_eval=False):
        if is_eval:
            all_prompts = [prompts for _ in rank_agents_list]
            all_labels = [labels for _ in rank_agents_list]
            all_metadata = [metadata for _ in rank_agents_list]
        else:
            if len(prompts) < len(rank_agents_list):
                raise ValueError("Number of prompts must be more than rank_agents_list")
            chunk_size = (len(prompts) + len(rank_agents_list) - 1) // len(rank_agents_list)
            all_prompts = [
                prompts[i*chunk_size: (i+1)*chunk_size] for i in range(len(rank_agents_list))]
            all_labels = [
                labels[i*chunk_size: (i+1)*chunk_size] for i in range(len(rank_agents_list))]
            all_metadata = [
                metadata[i*chunk_size: (i+1)*chunk_size] for i in range(len(rank_agents_list))]

        refs = []
        all_wrappers = []
        for per_llm_prompts, per_llm_labels, per_llm_metadata, rank_agents in zip(all_prompts, all_labels, all_metadata, rank_agents_list):
            multi_agent_wrapper = MultiAgentWrapper.remote(
                agents=rank_agents,
                workflow_args=self.workflow_args,
                workflow_func_path=self.args.workflow_func_path
            )
            # 2. The core change: Use the async wrapper to run workflows
            ref = multi_agent_wrapper.add_requests.remote(
                tool_manager=self.tool_manager,  # Pass your tool manager if you have one
                prompts=per_llm_prompts,
                labels=per_llm_labels,
                task=task,
                metadata=per_llm_metadata
            )
            refs.append(ref)
            all_wrappers.append(multi_agent_wrapper)
        ray.get(refs)

        all_output_refs = []
        for per_llm_prompts, wrapper in zip(all_prompts, all_wrappers):
            all_output_refs.append(wrapper.get_responses.remote(expected_len=len(per_llm_prompts)))
        all_trajectories = ray.get(all_output_refs)

        if is_eval:
            for trajectories in all_trajectories:
                assert len(trajectories) == len(
                    prompts), f"{len(trajectories)} vs {len(prompts)}"
            return all_trajectories
        else:
            all_trajectories = sum(all_trajectories, [])
            assert len(all_trajectories) == len(
                prompts), f"{len(all_trajectories)} vs {len(prompts)}"
            return all_trajectories

    def process_labels(self, labels):
        pass

    @torch.no_grad()
    def evaluate_samples(self, eval_data):
        args = self.strategy.args
        
        all_prompts, all_labels, all_metadata = eval_data[
            "prompt"], eval_data["label"], eval_data["metadata"]

        world_size = len(self.agents[0]["llms"])
        rank_agents_list = [self.get_rank_agent(
            rank=idx,
            world_size=world_size,
            is_eval=True) for idx in range(world_size)]

        if self.args.eval_workers > 0:
            rank_agents_list = rank_agents_list[:self.args.eval_workers]

        all_results = self.distribute_prompts(args.verify_task_eval,
                                              all_prompts,
                                              all_labels,
                                              all_metadata,
                                              rank_agents_list,
                                              is_eval=True)

        all_accuracies = [
            [trajectory["final_reward"] for trajectory in trajectories] for trajectories in all_results
        ]
        accuracy = np.mean([np.mean(acc) for acc in all_accuracies])

        return {"accuracy": accuracy, "metadata": all_results}
    
    @torch.no_grad()
    def generate_samples(self, all_prompts, rank=0, world_size=8):
        args = self.strategy.args
        # Set return_list to False, and then we only get one llm for async request
        rank_agents_list = [self.get_rank_agent(
            rank=rank,
            world_size=world_size,
            is_eval=False)]

        all_prompts, all_labels, all_metadata = all_prompts[
            "prompt"], all_prompts["label"], all_prompts["metadata"]
        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum(
            [[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum(
            [[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_metadata = sum(
            [[metadata] * args.n_samples_per_prompt for metadata in all_metadata], [])

        all_trajectories = self.distribute_prompts(args.verify_task,
                                              all_prompts,
                                              all_labels,
                                              all_metadata,
                                              rank_agents_list)

        training_samples = self.processor(all_trajectories, self.num_agents, self.args)

        if rank == 0:
            for index in range(3):
                for agent_index, agent_samples in enumerate(training_samples):
                    for key, values in agent_samples.items():
                        print(agent_index, key, str(values[index]))

        samples_list = [[] for _ in range(self.num_agents)]
        for agent_idx, samples in enumerate(training_samples):
            agent_tokenizer = self.agents[agent_idx]["tokenizer"]
            
            prompt_ids = [self.tokenize_fn_with_tok(prompt, agent_tokenizer, max_len=self.args.prompt_max_len) for prompt in samples["prompts"]]
            output_ids = [self.tokenize_fn_with_tok(output, agent_tokenizer, max_len=self.args.generate_max_len) for output in samples["outputs"]]
            all_labels = samples["labels"]

            for i in range(0, len(prompt_ids), args.micro_rollout_batch_size):
                prompts = prompt_ids[i: i + args.micro_rollout_batch_size]
                outputs = output_ids[i: i + args.micro_rollout_batch_size]
                labels = all_labels[i: i + args.micro_rollout_batch_size]

                samples_list[agent_idx].append(self.prepare_samples(
                    prompts=prompts,
                    outputs=outputs,
                    pred_labels=labels,
                    tokenizer=agent_tokenizer
                ))

        return {"sample": samples_list}

    def prepare_samples(self,
                        prompts,
                        outputs,
                        pred_labels,
                        action_mask_list=None,
                        num_agent_actions=None,
                        tokenizer=None):
        pred_labels = torch.tensor(
            pred_labels, device="cpu", dtype=torch.float)

        # NOTE: concat all outputs to following format:
        #
        # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
        # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
        pad_token_id, eos_token_id = tokenizer.pad_token_id, tokenizer.eos_token_id
        sequences = []
        packed_seq_lens = []
        attention_mask = []
        num_actions = []
        action_mask = []
        for i, output in enumerate(outputs):
            prompt = prompts[i]
            input_len = len(prompt)
            output_len = len(output)
            packed_seq_lens.append(input_len + output_len)
            sequences.extend(prompt + list(output))
            attention_mask.extend([i + 1] * (input_len + output_len))

            # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
            # num_actions.append(max(1, sum(current_action_mask)))
            assert output_len > 1, f"output_len = {output_len}"
            num_actions.append(max(1, output_len))
            action_mask.extend([1] * max(1, output_len))

        sequences = torch.tensor(sequences, device="cpu").unsqueeze(0)
        attention_mask = torch.tensor(
            attention_mask, device="cpu").unsqueeze(0)
        action_mask = torch.tensor(
            action_mask, device="cpu").unsqueeze(0)
        response_length = torch.tensor(
            num_actions, device="cpu", dtype=torch.float)
        total_length = torch.tensor(
            packed_seq_lens, device="cpu", dtype=torch.float)

        # if action_mask_list is not None:
        #     action_mask = sum(action_mask_list, [])
        #     assert len(action_mask) == sum(
        #         num_actions), f"action_mask ({len(action_mask)}) and num_actions ({sum(num_actions)}) should have the same length"
        #     # TODO: action_mask should be a int tensor not bool tensor
        #     action_mask = torch.tensor(
        #         action_mask, device="cpu", dtype=torch.int).unsqueeze(0)
        # else:
        #     action_mask = None

        # TODO: number of agents in each sample should keep consistent, so we save num_agent_actions in list rather than torch.tensor

        samples = Samples(
            sequences=sequences,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=num_actions,
            packed_seq_lens=packed_seq_lens,
            response_length=response_length,
            total_length=total_length,
            num_agent_actions=num_agent_actions,
            labels=pred_labels,
        )

        return samples
