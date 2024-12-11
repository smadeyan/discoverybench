import os
import json

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    # Load configurations
    model_config = load_config("model_config.json")
    api_config = load_config("api_config.json")
    
    # Get model details
    model_name = "gpt-4"  # or whatever model you want to use
    model_details = model_config["models"][model_name]
    api_key = api_config[model_details["model_type"]]
    
    # Create model interface
    model = create_model(
        api_type=model_details["model_type"],
        model_name=model_details["model_name"],
        api_key=api_key
    )
    
    # Create agent
    agent = Agent(
        llm=model,
        system_prompt="You are a discovery agent who can execute Python code to answer queries based on datasets.",
        max_iterations=25
    )
    
    # Run agent
    query = "Load the dataset from '/path/to/data.csv' and calculate the mean of column 'value'"
    result = agent.run(query, log_file="output.log")
    
    # Results are in result["steps"] and the notebook has been saved

if __name__ == "__main__":
    main()