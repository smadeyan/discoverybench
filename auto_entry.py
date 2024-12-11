# discovery_agent.py
import click
import json
import os
import asyncio
from typing import List
from agents.auto_react_agent import ReactAgent

def get_dv_query_for_real(metadata: dict, provide_domain_knowledge: bool, 
                         provide_workflow_tags: bool, nl_query: str) -> tuple[str, str]:
    """Enhanced query formatting with explicit dataset paths"""
    dataset_meta = ""
    for dataset_metadata in metadata['datasets']:
        dataset_meta += f"Dataset name: {dataset_metadata['name']}\n"
        dataset_meta += f"Dataset description: {dataset_metadata['description']}\n"
        dataset_meta += "Brief description of columns:\n"
        for col in dataset_metadata['columns']['raw']:
            dataset_meta += f"- {col['name']}: {col['description']}\n"
        dataset_meta += "\n"

    query_to_dv = dataset_meta

    # Add intermediate hypotheses if any
    if metadata.get('hypotheses', {}).get('intermediate'):
        query_to_dv += "\nIntermediate hypotheses:\n"
        for int_hypo in metadata['hypotheses']['intermediate']:
            query_to_dv += f"- {int_hypo['text']}\n"

    query_to_dv += f"\nQuery: {nl_query}\n"

    if provide_domain_knowledge and metadata.get('domain_knowledge'):
        query_to_dv += "\nDomain Knowledge:\n" + metadata['domain_knowledge'] + "\n"

    if provide_workflow_tags and metadata.get('workflow_tags'):
        query_to_dv += "\nMeta tags: " + metadata['workflow_tags'] + "\n"

    query_to_dv += ("\nIn your final answer, please write a scientific hypothesis in "
                   "natural language, derived from the provided dataset. Include:\n"
                   "1. Context of hypothesis\n"
                   "2. Variables chosen\n"
                   "3. Relationship between variables\n"
                   "4. Statistical significance\n\n"
                   "Also generate a summary of the full workflow starting from data loading that led to the final answer as WORKFLOW SUMMARY:")

    # Run the NL query through datavoyager
    print(f"query_to_dv: {query_to_dv}")
    return query_to_dv, dataset_meta

def get_dv_query_for_synth(metadata: dict, nl_query: str) -> tuple[str, str]:
    """Formats query for synthetic datasets"""
    dataset_meta = ""
    for dataset_metadata in metadata['datasets']:
        dataset_meta += "Dataset name: " + dataset_metadata['name']
        dataset_meta += "Dataset description: " + dataset_metadata['description']
        dataset_meta += "\nBrief description of columns: "
        for col in dataset_metadata['columns']:
            dataset_meta += col['name'] + ": " + col['description'] + ", "

    query_to_dv = dataset_meta

    query_to_dv += f"\nQuery: {nl_query}"
    query_to_dv += "In the final answer, please write down a scientific hypothesis in "\
        "natural language, derived from the provided dataset, clearly stating the "\
        "context of hypothesis (if any), variables chosen (if any) and "\
        "relationship between those variables (if any) including any statistical significance."

    # Run the NL query through datavoyager 
    print(f"query_to_dv: {query_to_dv}")
    return query_to_dv, dataset_meta

async def run_autonomous_single_agent_discoverybench(
    agent: ReactAgent,
    datasets: List[str],
    metadata: dict,
    nl_query: str,
    provide_domain_knowledge: bool,
    provide_workflow_tags: bool,
    dataset_type: str
) -> dict:
    """Runs the discovery agent with appropriate query formatting"""
    if dataset_type == "real":
        query_to_dv, dataset_meta = get_dv_query_for_real(
            metadata,
            provide_domain_knowledge,
            provide_workflow_tags,
            nl_query
        )
    else:
        query_to_dv, dataset_meta = get_dv_query_for_synth(
            metadata,
            nl_query
        )

    return await agent.generate(
        dataset_paths=datasets,
        query=query_to_dv
    )

@click.command()
@click.option('--model_name', default='gpt-4', help='Model name from config')
@click.option('--api_config', default='config/api_config.json', help='API config file')
@click.option('--log_file', default='discovery_agent.log', help='Log file')
@click.option('--notebook_output', default='notebooks/output.ipynb', help='Output notebook path')
@click.option('--metadata_path', required=True, help='Metadata file path')
@click.option('--metadata_type', type=click.Choice(['real', 'synth']), required=True)
@click.option('--add_domain_knowledge', is_flag=True)
@click.option('--add_workflow_tags', is_flag=True)
@click.option('--use_reflection', is_flag=True)
@click.argument('query')
def main(
    query: str,
    model_name: str,
    api_config: str,
    log_file: str,
    notebook_output: str,
    metadata_path: str,
    metadata_type: str,
    add_domain_knowledge: bool,
    add_workflow_tags: bool,
    use_reflection: bool
):
    notebook_dir = os.path.dirname(notebook_output)
    if notebook_dir:
        os.makedirs(notebook_dir, exist_ok=True)
    
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    metadata_dir = os.path.dirname(metadata_path)
    dataset_paths = [
        os.path.join(metadata_dir, dataset['name'])
        for dataset in metadata['datasets']
    ]

    agent = ReactAgent(
        model_name=model_name,
        api_config=api_config,
        log_file=log_file,
        use_reflection=use_reflection
    )

    result = asyncio.run(run_autonomous_single_agent_discoverybench(
        agent=agent,
        datasets=dataset_paths,
        metadata=metadata,
        nl_query=query,
        provide_domain_knowledge=add_domain_knowledge,
        provide_workflow_tags=add_workflow_tags,
        dataset_type=metadata_type
    ))

    agent.save_notebook(notebook_output)

    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()