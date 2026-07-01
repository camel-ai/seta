import os
import asyncio
from typing import Any, Dict
from pipeline_base import AnalysisAgent, EvolAgent, TaskContext, AnalysisResult, EvolvedTask
from io_utils import get_next_version_path, load_cleaned_trajectory

try:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    from ..evol_instruct import EvolInstructPipeline
except ImportError:
    print("Warning: camel.agents or local evol_instruct not found.")

class CamelAnalysisAgent(AnalysisAgent):
    def __init__(self, model_platform: str = "openai", model_type: str = "gpt-4o"):
        self.system_message = BaseMessage.make_assistant_message(
            role_name="Analysis Agent",
            content="You are an expert coding agent analyzer. You analyze execution traces to identify failures and suggest improvements."
        )
        try:
            from camel.models import ModelFactory
            from camel.types import ModelPlatformType, ModelType
            # Note: This is an example init, you might want to pass more configs
            self.agent = ChatAgent(
                system_message=self.system_message,
            )
        except Exception as e:
            print(f"Failed to init Camel ChatAgent: {e}")
            self.agent = None

    async def analyze(self, task_context: TaskContext, **kwargs) -> AnalysisResult:
        task_id = task_context.task_id
        
        if not self.agent:
            return AnalysisResult(task_id, "Error: Camel Agent not initialized.")
            
        # Enforce output dir to be next version (e.g. seed_v1/{task_id})
        output_dir = get_next_version_path(task_context.metadata.get("seed_path", ""))
        os.makedirs(output_dir, exist_ok=True)
        
        log_file_path = os.path.join(output_dir, "camel_analysis_agent_log.txt")
        
        # Format rollout info for batch analysis
        rollout_info_lines = []
        for i, r in enumerate(task_context.rollouts):
            rollout_info_lines.append(f"--- Rollout {i} ---")
            cleaned_traj = load_cleaned_trajectory(r['trajectory'])
            rollout_info_lines.append(f"Cleaned Trajectory:\n{cleaned_traj}")
            rollout_info_lines.append(f"Test Results Path: {r['test_results']}")
            rollout_info_lines.append("")
        
        rollout_info = "\n".join(rollout_info_lines)
        
        # Load external prompt
        prompt_path = os.path.join(os.path.dirname(__file__), "analysis_agent_prompt.md")
        with open(prompt_path, 'r') as f:
            prompt_tmpl = f.read()
        
        user_msg = prompt_tmpl.format(rollout_info=rollout_info)
        
        loop = asyncio.get_running_loop()
        
        def _run_step():
            self.agent.reset()
            response = self.agent.step(user_msg)
            return response.msgs[0].content
            
        print(f"[{task_id}] Camel Analysis log: {log_file_path}")
        analysis_content = await loop.run_in_executor(None, _run_step)
        
        # Write log manually for Camel if needed, or if ChatAgent doesn't do it automatically
        with open(log_file_path, 'w') as f:
            f.write(f"PROMPT:\n{user_msg}\n\nRESPONSE:\n{analysis_content}")
        
        return AnalysisResult(
            task_id=task_id,
            analysis_content=analysis_content,
            metadata={"output_dir": output_dir, "log_file": log_file_path}
        )

class CamelEvolAgent(EvolAgent):
    def __init__(self):
        # reuse the existing EvolInstructPipeline logic
        self.pipeline = EvolInstructPipeline()

    async def evolve(self, task_context: TaskContext, analysis_result: AnalysisResult, **kwargs) -> EvolvedTask:
        task_id = task_context.task_id
        
        # Enforce output dir to be next version (e.g. seed_v1/{task_id})
        evol_cwd = get_next_version_path(task_context.metadata.get("seed_path", ""))
        os.makedirs(evol_cwd, exist_ok=True)
        
        log_file_path = os.path.join(evol_cwd, "camel_evol_agent_log.txt")
        
        # Load external prompt
        prompt_path = os.path.join(os.path.dirname(__file__), "evol_agent_prompt.md")
        with open(prompt_path, 'r') as f:
            prompt_tmpl = f.read()

        user_msg = prompt_tmpl.format(
            task_id=task_id, 
            analysis_content=analysis_result.analysis_content
        )
        
        loop = asyncio.get_running_loop()
        
        def _run_evol():
            # Note: EvolInstructPipeline might need adjustments if it expects single instruction
            # but here we pass the formatted persona prompt.
            results = self.pipeline.generate(
                prompts=[user_msg],
                evolution_spec="deepening", 
                num_generations=1
            )
            
            if results and len(results) > 0:
                prompt_results = results[0]
                last_iter = max(prompt_results.keys())
                candidates = prompt_results[last_iter]
                if candidates:
                    return candidates[0]["instruction"]
            return "Evolution failed."

        print(f"[{task_id}] Camel Evol log: {log_file_path}")
        new_instruction = await loop.run_in_executor(None, _run_evol)
        
        # Write log
        with open(log_file_path, 'w') as f:
            f.write(f"ANALYSIS:\n{analysis_result.analysis_content}\n\nEVOLVED INSTRUCTION:\n{new_instruction}")
            
        # Write the actual file to the new versioned folder
        with open(os.path.join(evol_cwd, "task_instruction_evolved.txt"), 'w') as f:
            f.write(new_instruction)
        
        return EvolvedTask(
            task_id=task_id,
            new_files={"task_instruction_evolved.txt": new_instruction},
            metadata={"evol_cwd": evol_cwd, "log_file": log_file_path}
        )
