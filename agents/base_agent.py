import openai
from openai import AsyncOpenAI
import json
import os
import logging
import re
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_output
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import ast
import sys
from io import StringIO

@dataclass
class AgentAction:
    tool: str
    tool_input: str
    log: str = ""

@dataclass
class AgentFinish:
    return_values: dict
    log: str = ""

class CodeExecutionTool:
    def __init__(self):
        self.locals = {}
        self.globals = {}
    
    def execute(self, code: str) -> str:
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output

        try:
            ast.parse(code)
            exec(code, self.globals, self.locals)
            output = redirected_output.getvalue()
            lines = code.strip().split('\n')
            if lines:
                try:
                    last_line = lines[-1]
                    if not any(last_line.startswith(x) for x in ['def ', 'class ', 'if ', 'for ', 'while ', '#']):
                        result = eval(last_line, self.globals, self.locals)
                        if result is not None:
                            output += str(result)
                except:
                    pass
            
            return output
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            sys.stdout = old_stdout

class Logger:
    def __init__(self, model_name: str, log_file: str):
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, log_file)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"{model_name}")
    
    def log_action(self, action_type: str, content: str):
        self.logger.info(f"=== {action_type} ===\n{content}\n")
    
    def log_error(self, error: str):
        self.logger.error(f"=== ERROR ===\n{error}\n")

class BaseDiscoveryAgent:
    def __init__(
        self,
        model_name: str,
        api_config: str,
        log_file: str,
        max_iterations: int = 10,
        temperature: float = 0.7
    ):
        with open(api_config, 'r') as f:
            api_config_data = json.load(f)
            
        with open('config/model_config.json', 'r') as f:
            model_config = json.load(f)
            
        try:
            model_details = model_config['models'][model_name]
            self.model = model_details['model_name']
            self.model_type = model_details['model_type']
        except KeyError:
            raise ValueError(f"Model {model_name} not found in model config")
            
        try:
            openai.api_key = api_config_data[self.model_type]
        except KeyError:
            raise ValueError(f"API key not found for {self.model_type}")

        self.max_iterations = max_iterations
        self.temperature = temperature
        self.api_key = api_config_data[self.model_type]
        
        self.logger = Logger(model_name, log_file)
        self.code_executor = CodeExecutionTool()
        self.notebook_cells = []

    async def _get_completion(self, messages: List[Dict[str, str]]) -> str:
        print("+++++", self.api_key)
        client = AsyncOpenAI(api_key=self.api_key, base_url="https://cmu.litellm.ai")
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.log_error(f"API call failed: {str(e)}")
            raise

    def add_notebook_cell(self, code: str, output: str = None):
        cell = new_code_cell(source=code)
        
        if output:
            output_obj = new_output(
                output_type='stream',
                name='stdout',
                text=str(output)
            )
            cell.outputs = [output_obj]
        
        self.notebook_cells.append(cell)
        
    def save_notebook(self, path: str):
        nb = new_notebook(cells=self.notebook_cells)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            

    async def generate(self, dataset_paths: List[str], query: str) -> dict:
        raise NotImplementedError