# Set up the base template
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.tools import BaseTool
from langchain.chains.llm import LLMChain
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from typing import List, Union, Dict, Any, Tuple, Sequence, Optional
from langchain.agents.agent import BaseSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain.schema.runnable import RunnableSequence
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.callbacks.base import Callbacks
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
import re
from pydantic import Field, PrivateAttr
import logging
from datetime import datetime
import os

reflection_template = """
You are a reflective assistant. Review the previous steps and provide constructive feedback:

Previous steps:
{previous_steps}

Reflect on the following:
1. Were the actions taken appropriate and efficient?
2. Is there a better approach to solve the problem?
3. What improvements can be made in the reasoning process?

Provide your reflection and suggestions for improvement:
"""

# Define the reflection prompt template
reflection_prompt_template = """You are a reflection assistant.

You have the following thoughts, actions, and observations so far:

{scratchpad}

Reflect on these and determine whether the last action was appropriate and effective in moving towards the final answer.

Your reflection should be concise and help guide the next steps.

Reflection:"""

template = """{system_prompt}

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Result Evaluation: you should always evaluate whether the results are as expected (e.g., are the number and rows in the data frame what was expected, or is the data type as expected, etc).
Reflection: you should always reflect on whether this action moved you closer to the final answer or if a new approach is needed.
... (this Thought/Action/Action Input/Observation/Result Evaluation/Reflection can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. In the final answer, please write down a scientific hypothesis in natural language, derived from the provided dataset, clearly stating the context of hypothesis (if any), variables chosen (if any) and relationship between those variables (if any) including any statistical significance. If available, cite any additional figures to back your hypothesis. Also generate a summary of the full workflow starting from data loading that led to the final answer as WORKFLOW SUMMARY:


Example
Question: Load all datasets using python using provided paths. Paths: /Users/bodhi/projects/datavoyager/DiscoveryBench/evolution_freshwater_fish/body-size-evolution-in-south-american-freshwater-fishes.csv. What is the linear coefficient that describes the positive relationship between the rate of maximum body length evolution and spatial variation in speciation rates, where the former emerges as the most influential factor?
Thought: First, I need to load the dataset from the provided path using Python, and then analyze the data to find the linear coefficient that describes the relationship between the rate of maximum body length evolution and spatial variation in speciation rates.
Action: python_repl_ast
Action Input: import pandas as pd\n\ndf = pd.read_csv('/Users/bodhi/projects/datavoyager/DiscoveryBench/evolution_freshwater_fish/body-size-evolution-in-south-american-freshwater-fishes.csv')\ndf.head()
Observation: 
     HYBAS_ID       long       lat  BAMM_speciation  BAMM_extinction  BAMM_NetDiv  ...   aet    Elevation  sgr  soil_div     area  diversity
0  6050000010 -76.477422  7.742693         0.137392         0.026807     0.110585  ...  1387   330.150088  166  0.482402  72363.7         68
1  6050000740 -74.628725  9.803586         0.117235         0.025796     0.091438  ...  1082    69.475294   23  0.457436  17944.3         35
2  6050068100 -75.295995  8.448815         0.119381         0.023826     0.095555  ...  1312   143.032178   74  0.378793  17105.5         44
3  6050068110 -74.608408  8.922863         0.132477         0.027777     0.104700  ...  1445    14.724138    3  0.468328    610.1         48
4  6050070260 -75.591588  5.770093         0.120127         0.022940     0.097187  ...  1371  1378.729945  421  0.158870  61901.9         81
[5 rows x 21 columns]
Result Evaluation: The data has loaded correctly, and the first few rows display as expected with the necessary columns present.
Reflection: Loading the data was successful, and I can proceed to the next step of analysis.
Thought: Now that the dataset is loaded, I need to perform a linear regression analysis to determine the coefficient that describes the relationship between the rate of maximum body length evolution ('BAMM_speciation') and spatial variation in speciation rates ('BAMM_NetDiv').
Action: python_repl_ast
Action Input: from sklearn.linear_model import LinearRegression\nimport numpy as np\n\n# Prepare the data for regression\nX = df[['BAMM_speciation']].values.reshape(-1, 1)  # Predictor\ny = df['BAMM_NetDiv'].values  # Response\n\n# Create a linear regression model\nmodel = LinearRegression()\nmodel.fit(X, y)\n\n# Get the coefficient\ncoefficient = model.coef_[0]\ncoefficient
Observation: 0.5175306498596297
Result Evaluation: The output is a numerical coefficient, which seems reasonable for a linear relationship coefficient.
Reflection: I have obtained the coefficient needed to answer the question.
WORKFLOW_SUMMARY:
1. Data Loading: Loaded the dataset from the specified path using Python.
2. Data Inspection: Displayed the first few rows of the dataset to understand its structure and the relevant columns.
3. Linear Regression Analysis: Performed a linear regression analysis using 'BAMM_speciation' as the predictor and 'BAMM_NetDiv' as the response variable to find the linear coefficient.
FINAL_ANSWER:
The linear coefficient that describes the positive relationship between the rate of maximum body length evolution ('BAMM_speciation') and spatial variation in speciation rates ('BAMM_NetDiv') is approximately 0.518.


Begin!

Question: {input}
Your previous thoughts, observations and reflections, if any, are given in the below scratchpad:
##Scratchpad start##
{agent_scratchpad}
##Scratchpad end##"""


reflection_template = """{system_prompt}

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Result Evaluation: Evaluate your observation/result and check whether the correct data columns were used, and if the results are as expected (e.g., are the number and rows in the data frame what was expected, or is the data type as expected, etc).
Reflection: Reflect on whether this action, and previous actions in scratchpad, have moved you closer to the final answer or if a new approach is needed.
... (this Thought/Action/Action Input/Observation/Result Evaluation/Reflection can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. In the final answer, please write down a scientific hypothesis in natural language, derived from the provided dataset, clearly stating the context of hypothesis (if any), variables chosen (if any), and relationship between those variables (if any), including any statistical significance. Also generate a summary of the full workflow starting from data loading that led to the final answer as WORKFLOW SUMMARY:

Example
Question: Load all datasets using python using provided paths. Paths: /Users/bodhi/projects/datavoyager/DiscoveryBench/evolution_freshwater_fish/body-size-evolution-in-south-american-freshwater-fishes.csv. What is the linear coefficient that describes the positive relationship between the rate of maximum body length evolution and spatial variation in speciation rates, where the former emerges as the most influential factor?
Thought: First, I need to load the dataset from the provided path using Python, and then analyze the data to find the linear coefficient that describes the relationship between the rate of maximum body length evolution and spatial variation in speciation rates.
Action: python_repl_ast
Action Input: import pandas as pd\n\ndf = pd.read_csv('/Users/bodhi/projects/datavoyager/DiscoveryBench/evolution_freshwater_fish/body-size-evolution-in-south-american-freshwater-fishes.csv')\ndf.head()
Observation: 
     HYBAS_ID       long       lat  BAMM_speciation  BAMM_extinction  BAMM_NetDiv  ...   aet    Elevation  sgr  soil_div     area  diversity
0  6050000010 -76.477422  7.742693         0.137392         0.026807     0.110585  ...  1387   330.150088  166  0.482402  72363.7         68
1  6050000740 -74.628725  9.803586         0.117235         0.025796     0.091438  ...  1082    69.475294   23  0.457436  17944.3         35
2  6050068100 -75.295995  8.448815         0.119381         0.023826     0.095555  ...  1312   143.032178   74  0.378793  17105.5         44
3  6050068110 -74.608408  8.922863         0.132477         0.027777     0.104700  ...  1445    14.724138    3  0.468328    610.1         48
4  6050070260 -75.591588  5.770093         0.120127         0.022940     0.097187  ...  1371  1378.729945  421  0.158870  61901.9         81
[5 rows x 21 columns]
...
Result Evaluation: The data has loaded correctly, and the first few rows display as expected with the necessary columns present.
Reflection: Loading the data was successful, and I can proceed to the next step of analysis.
Thought: Now that the dataset is loaded, I need to perform a linear regression analysis to determine the coefficient that describes the relationship between 'BAMM_speciation' and 'BAMM_NetDiv'.
Action: python_repl_ast
Action Input: from sklearn.linear_model import LinearRegression\nimport numpy as np\n\n# Prepare the data for regression\nX = df[['BAMM_speciation']].values.reshape(-1, 1)  # Predictor\ny = df['BAMM_NetDiv'].values  # Response\n\n# Create a linear regression model\nmodel = LinearRegression()\nmodel.fit(X, y)\n\n# Get the coefficient\ncoefficient = model.coef_[0]\ncoefficient
Observation: 0.5175306498596297
Result Evaluation: The output is a numerical coefficient, which seems reasonable for a linear relationship coefficient.
Reflection: I have obtained the coefficient needed to answer the question.
Thought: I now know the final answer
Final Answer:
The linear coefficient that describes the positive relationship between the rate of maximum body length evolution ('BAMM_speciation') and spatial variation in speciation rates ('BAMM_NetDiv') is approximately 0.518.

WORKFLOW SUMMARY:
1. Data Loading: Loaded the dataset from the specified path using Python.
2. Data Inspection: Displayed the first few rows of the dataset to confirm successful loading.
3. Linear Regression Analysis: Performed a linear regression analysis using 'BAMM_speciation' as the predictor and 'BAMM_NetDiv' as the response variable to find the linear coefficient.

Begin!

Question: {input}
Your previous thoughts and observations, if any, are given in the below scratchpad:
##Scratchpad start##
{agent_scratchpad}
##Scratchpad end##"""

class CustomLogger:
    def __init__(self, log_dir="logs"):
        # Create logs directory if it doesn't exist
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a unique log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"agent_log_{timestamp}.txt")
        
        # Set up logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # This will also print to console
            ]
        )
        
        # Create logger instance
        self.logger = logging.getLogger("AgentLogger")
        
    def log_action(self, action_type, content):
        """Log an action with a specific type"""
        self.logger.info(f"=== {action_type} ===\n{content}\n")
        
    def log_observation(self, observation):
        """Log an observation"""
        self.logger.info(f"=== OBSERVATION ===\n{observation}\n")
        
    def log_reflection(self, reflection):
        """Log a reflection"""
        self.logger.info(f"=== REFLECTION ===\n{reflection}\n")
        
    def log_error(self, error):
        """Log an error"""
        self.logger.error(f"=== ERROR ===\n{error}\n")

# Create a global logger instance
agent_logger = CustomLogger()

class NotebookCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.code_cells = []

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        if action.tool == "python_repl_ast":
            code_snippet = action.tool_input
            self.code_cells.append(new_code_cell(code_snippet))

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            # if action.tool == "Reflection":
            #     thoughts += f"Reflection: {observation}\n"
            # else:
            #     thoughts += action.log
            #     thoughts += f"\nObservation: {observation}\nThought: "
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
            # # Add a Reflection step here
            # thoughts += f"\nReflection: Based on the last observation, consider whether the action moved closer to a solution or if an alternative approach is needed."
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

# CustomOutputParser to parse the output of the LLM and execute actions
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "final answer:" in llm_output.lower():
            agent_logger.log_action("FINAL ANSWER", llm_output.split("Final Answer:")[-1].strip())
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        agent_logger.log_action("RAW OUTPUT", llm_output)

        # regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        # match = re.search(regex, llm_output, re.DOTALL)
        # if not match:
        #     raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        # action = match.group(1).strip()
        # action_input = match.group(2)

        # # Return the action and action input
        # agent_logger.log_action("PARSED ACTION", f"Action: {action}\nInput: {action_input.strip()}")
        # return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        # Updated regex pattern to include Result Evaluation and Reflection
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*?)\nObservation:\s*(.*?)\nResult Evaluation:\s*(.*?)\nReflection:\s*(.*?)(?=\n\w|$)"
        
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            error_msg = f"Could not parse LLM output: `{llm_output}`"
            agent_logger.log_error(error_msg)
            raise ValueError(error_msg)
            
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        observation = match.group(3).strip()
        evaluation = match.group(4).strip()
        reflection = match.group(5).strip()
        
        # Log all components
        agent_logger.log_action("PARSED COMPONENTS", 
            f"Action: {action}\n"
            f"Input: {action_input}\n"
            f"Observation: {observation}\n"
            f"Evaluation: {evaluation}\n"
            f"Reflection: {reflection}")
        
        return AgentAction(
            tool=action,
            tool_input=action_input.strip(" ").strip('"'),
            log=llm_output
        )


def create_agent(
    llm,
    handlers,
    max_iterations = None,
    use_reflection = False,
    early_stopping_method: str = "force",
):
    output_parser = CustomOutputParser()
    python_tool = PythonAstREPLTool(callbacks=handlers)
    tools = [python_tool]
    tool_names = [tool.name for tool in tools]

    template_to_use = template
    # if use_reflection:
    #     template_to_use = reflection_template

    prompt_main = CustomPromptTemplate(
        template=template_to_use,
        tools=tools,
        input_variables=["system_prompt", "input", "intermediate_steps"]
    )

    # Reflection prompt
    prompt_reflection = PromptTemplate(
        template=reflection_template,
        input_variables=["previous_steps"],
    )

    llm_chain_main = LLMChain(llm=llm, prompt=prompt_main, callbacks=handlers)
    reflection_chain = LLMChain(llm=llm, prompt=prompt_reflection)

#     reflection_prompt = PromptTemplate(
#         input_variables=["agent_scratchpad"],
#         template="""You are now in the reflection step.

# Your previous thoughts, actions, and observations are as follows:
# {agent_scratchpad}

# Reflect on whether the results are as expected (e.g., are the number of rows in the data frame what was expected, is the data type as expected, etc.). 

# Provide a concise reflection considering the observations.

# Reflection:"""
#     )

#     reflection_chain = LLMChain(llm=llm, prompt=reflection_prompt, callbacks=handlers)

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain_main,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    # if use_reflection:
    #     print(type(tool_names))
    #     agent = AlternatingAgent(
    #     llm_chain_main=llm_chain_main,
    #     llm_chain_reflection=llm_chain_reflection,
    #     output_parser=output_parser,
    #     allowed_tools=tool_names,
    #     stop=["\nObservation:"],
    # )

    

    class ReflectiveAgentExecutor(AgentExecutor):
        def _take_next_step(
            self,
            name_to_tool_map: Dict[str, BaseTool],
            color_mapping: Dict[str, str],
            inputs: Dict[str, str],
            intermediate_steps: List[Tuple[AgentAction, str]],
            run_manager: Optional[CallbackManagerForChainRun] = None,
        ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
            # Perform the main agent step
            output = super()._take_next_step(
                name_to_tool_map, color_mapping, inputs, intermediate_steps, run_manager
            )
            
            # If the agent has finished, return the result
            if isinstance(output, AgentFinish):
                return output
            
            # Otherwise, perform reflection
            previous_steps = "\n".join([f"{action.log}\nObservation: {observation}" for action, observation in intermediate_steps])
            reflection = reflection_chain.run(previous_steps=previous_steps)
            
            # Add the reflection as an observation to the intermediate steps
            reflection_observation = f"Reflection: {reflection}"
            intermediate_steps.append((AgentAction(tool="Reflection", tool_input="", log=""), reflection_observation))
            
            return intermediate_steps

    if use_reflection:
        return ReflectiveAgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=max_iterations
        )

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_iterations,
        callbacks=handlers,
        early_stopping_method=early_stopping_method
    )


class AlternatingAgent(BaseSingleActionAgent):
    llm_chain_main: Any = Field(...)
    llm_chain_reflection: Any = Field(...)
    output_parser: Any = Field(...)
    # allowed_tools: List[str]
    stop: Union[List[str], str] = Field(default_factory=list)
    reflection: str = Field(default="")
    _allowed_tools: List[str] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        llm_chain_main,
        llm_chain_reflection,
        output_parser,
        allowed_tools,
        stop,
    ):
        super().__init__(
            llm_chain_main=llm_chain_main,
            llm_chain_reflection=llm_chain_reflection,
            output_parser=output_parser,
            _allowed_tools = allowed_tools,
            stop=stop
        )
        self.reflection = ""

    class Config:
        arbitrary_types_allowed = True
    
    @property
    def input_keys(self) -> List[str]:
        return ["input", "system_prompt"]
    
    @property
    def allowed_tools(self) -> List[str]:
        return self._allowed_tools

    @allowed_tools.setter
    def allowed_tools(self, tools: List[str]):
        self._allowed_tools = tools

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        return self._plan(intermediate_steps, callbacks, **kwargs)
    
    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        return await self._plan(intermediate_steps, callbacks, **kwargs)

    def _plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        if len(intermediate_steps) == 0 or len(intermediate_steps) % 2 == 0:
            inputs = {
                "system_prompt": kwargs.get("system_prompt", ""),
                "input": kwargs.get("input", ""),
                "intermediate_steps": intermediate_steps,
                "reflection": self.reflection,
            }
            llm_chain = self.llm_chain_main
        else:
            scratchpad = self.construct_scratchpad(intermediate_steps)
            inputs = {"scratchpad": scratchpad}
            llm_chain = self.llm_chain_reflection

        output = llm_chain.run(inputs)

        if llm_chain == self.llm_chain_reflection:
            self.reflection = output.strip()
            return self.plan(intermediate_steps, **kwargs)
        else:
            return self.output_parser.parse(output)

    def construct_scratchpad(self, intermediate_steps):
        scratchpad = ""
        for action, observation in intermediate_steps:
            scratchpad += action.log
            scratchpad += f"\nObservation: {observation}\n"
        return scratchpad