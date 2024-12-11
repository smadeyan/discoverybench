# react_agent.py
from agents.base_agent import BaseDiscoveryAgent, AgentAction, AgentFinish
from nbformat.v4 import new_code_cell, new_output, new_markdown_cell
from typing import List, Dict, Any, Optional, Union
import re
import os

class ReactAgent(BaseDiscoveryAgent):
    """Implementation of a React-style agent"""
    
    def __init__(self, model_name: str, api_config: str, log_file: str, use_reflection: bool = False):
        super().__init__(model_name, api_config, log_file)
        self.use_reflection = use_reflection
        self.system_prompt = self._load_system_prompt()
        self.iteration_count = 0
        self.dataset_paths = []
        self.scratchpad = []

    def _extract_code_from_output(self, output: str) -> List[str]:
        code_blocks = []
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, output, re.DOTALL)
        return [m.strip() for m in matches]
    
    def _create_output_object(self, output_text: str):
        if output_text:
            return new_output(
                output_type='stream',
                name='stdout',
                text=str(output_text)
            )

    def _load_system_prompt(self) -> str:
        return """You are a discovery agent who can execute Python code to answer queries based on datasets.
        
        Important:
        - You will be provided with dataset paths - always use these exact paths when loading data
        - Provide ONE action at a time and wait for its result before proceeding
        - Do not assume or predict the results of code execution
        - Only proceed to the next step after receiving and evaluating the observation
        - Review the scratchpad to understand previous steps and their outcomes
        
        Use this format for each step:
        Thought: think about what to do next based on previous observations and scratchpad
        Action: python_repl_ast
        Action Input: a single piece of Python code to execute
        
        After receiving an observation, use this format:
        Result Evaluation: evaluate if the results match expectations or if there were errors
        Reflection: reflect on whether this helped answer the question
        
        Only when you have enough information to answer the question:
        Thought: I now know the final answer
        Final Answer: your final answer including hypothesis and workflow summary
        """
    
    def _format_scratchpad(self) -> str:
        if not self.scratchpad:
            return "No previous steps recorded."
            
        formatted = "=== ANALYSIS HISTORY ===\n"
        for i, step in enumerate(self.scratchpad, 1):
            formatted += f"\nStep {i}:\n"
            for key, value in step.items():
                formatted += f"{key}:\n{value}\n"
        return formatted

    def _add_to_scratchpad(self, step_dict: dict):
        self.scratchpad.append(step_dict)
    
    def _create_initial_message(self, query: str) -> dict:
        dataset_info = "\n".join([f"Dataset path {i+1}: {path}" for i, path in enumerate(self.dataset_paths)])
        scratchpad = self._format_scratchpad()
        
        content = (
            f"Available dataset paths:\n{dataset_info}\n\n"
            f"Query: {query}\n\n"
            f"\nScratchpad:\n{scratchpad}\n\n"
            "Important: Provide only ONE action at a time. Wait for the observation "
            "before proceeding to the next step. Do not assume results."
        )
        return {"role": "user", "content": content}

    def _evaluate_observation(self, observation: str) -> bool:
        error_indicators = ["Error:", "Exception:", "No such file", "undefined"]
        return not any(indicator in observation for indicator in error_indicators)

    def _add_code_to_notebook(self, code: str, exec_output: str, successful: bool = True) -> None:
        if not successful:
            status_cell = new_markdown_cell(source="**Note**: The following code execution was unsuccessful")
            self.notebook_cells.append(status_cell)
        
        cell = new_code_cell(source=code)
        if exec_output:
            output_obj = self._create_output_object(exec_output)
            if output_obj:
                cell.outputs = [output_obj]
        self.notebook_cells.append(cell)


    def _parse_output(self, output: str) -> Union[AgentAction, AgentFinish]:
        if "final answer:" in output.lower():
            self.logger.log_action("FINAL_ANSWER", output)
            return AgentFinish(
                return_values={"output": output.split("Final Answer:")[-1].strip()},
                log=output
            )

        action_match = re.search(
            r"Action:\s*(.*?)[\n\r]+Action Input:[\s\n\r]*((?:(?!Action:)[\s\S])*)",
            output,
            re.IGNORECASE
        )

        print("DEBUG: Full output:", output)

        if not action_match:
            if re.search(r"Thought:.*?(?:Thought:|Action:)", output, re.DOTALL):
                raise ValueError("Multiple steps detected. Please provide one action at a time.")
            raise ValueError(f"Could not parse output: {output}")

        tool = action_match.group(1).strip()
        raw_input = action_match.group(2).strip()

        code_block_match = re.search(r"```(?:python)?\n(.*?)\n```", raw_input, re.DOTALL)
        tool_input = code_block_match.group(1).strip() if code_block_match else raw_input.strip('`').strip()

        print("DEBUG: Extracted tool:", tool)
        print("DEBUG: Raw input:", raw_input)
        print("DEBUG: Final tool input:", tool_input)

        return AgentAction(
            tool=tool,
            tool_input=tool_input,
            log=output
        )

    async def generate(self, dataset_paths: List[str], query: str) -> dict:
        self.dataset_paths = [os.path.abspath(path) for path in dataset_paths]
        messages = [
            {"role": "system", "content": self.system_prompt},
            self._create_initial_message(query)
        ]

        self.iteration_count = 0
        
        for _ in range(self.max_iterations):
            try:
                self.iteration_count += 1
                output = await self._get_completion(messages)
                self.logger.log_action("LLM_OUTPUT", output)
                
                next_step = self._parse_output(output)
                
                if isinstance(next_step, AgentFinish):
                    self._add_to_scratchpad({
                        "Final Thought": "I now know the final answer",
                        "Final Answer": next_step.return_values["output"]
                    })
                    
                    self.logger.log_action("FINAL_STATS", f"Completed in {self.iteration_count} iterations")
                    return {
                        **next_step.return_values,
                        "iterations": self.iteration_count,
                        "analysis_history": self._format_scratchpad()
                    }
                
                if next_step.tool == "python_repl_ast":
                    print("+++Inside code conditional")
                    code = next_step.tool_input
                    print("+++code ", code)
                    exec_output = self.code_executor.execute(code)

                    self.add_notebook_cell(code, exec_output)
                    
                    current_step = {
                        "Thought": output.split("Thought:")[1].split("Action:")[0].strip(),
                        "Action": "python_repl_ast",
                        "Code": code,
                        "Observation": exec_output
                    }

                    print("++++", current_step)
                    
                    if "Result Evaluation:" in output:
                        current_step["Result Evaluation"] = output.split("Result Evaluation:")[1].split("Reflection:")[0].strip()
                    if "Reflection:" in output:
                        current_step["Reflection"] = output.split("Reflection:")[1].strip()
                    
                    self._add_to_scratchpad(current_step)
                    
                    observation = f"Observation: {exec_output}"
                else:
                    observation = f"Error: Unknown tool {next_step.tool}"
                    self._add_to_scratchpad({
                        "Error": observation
                    })
                
                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": f"{observation}\n\nCurrent Analysis Status:\n{self._format_scratchpad()}"})
                
            except ValueError as e:
                error_msg = str(e)
                self._add_to_scratchpad({
                    "Error": error_msg
                })
                messages.append({"role": "assistant", "content": output})
                messages.append({
                    "role": "user", 
                    "content": f"Error: {error_msg} Please provide only one action and wait for its result.\n\nCurrent Analysis Status:\n{self._format_scratchpad()}"
                })
                continue
                
            except Exception as e:
                self.logger.log_error(str(e))
                return {"error": str(e), "iterations": self.iteration_count}
        
        return {"error": "Max iterations reached", "iterations": self.iteration_count}