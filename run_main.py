import argparse
from model import Generator, Revision
import numpy as np
import random
import asyncio
from pathlib import Path

base_dir = Path(__file__).parent

def get_parser(train_flag, g_llm_model, r_llm_model, g_temperature, g_variants_per_problem, dataset):
    parser = argparse.ArgumentParser(description='LLM-OM')
    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='LLM-OM', help='task name')
    parser.add_argument('--train_flag', type=str, required=False, default=train_flag, 
                        help='Set different experiments, options: [IO, CoT, main] [IO mode, CoT mode, main experiment]')
    
    # path
    parser.add_argument('--cache_path', type=str, required=False, default=f"{base_dir}/cache", help="cache path")
    parser.add_argument('--data_path', type=str, required=False, default=f"{base_dir}/data", help="dataset path")
    parser.add_argument('--output_path', type=str, required=False, default=f"{base_dir}/output", help="output path")
    parser.add_argument('--dataset',type=str, required=False, default=dataset, help="dataset name, options: [1, 2, complexor, industryor_easy, industryor_medium, industryor_hard, nl4opt, nlp4lp]")

    # generator_llm_model
    parser.add_argument('--g_llm_model',type=str, required=False, default=g_llm_model, help="LLM model name, options: [gpt-4o-mini, claude-3-7-sonnet-20250219, deepseek-r1-250528, o4-mini]")
    parser.add_argument('--g_api_key',type=str, required=False, default="sk-sJxxxxxxxn4wgZlMlYTUBiNbGYIUWoqCEPAu1Jwdm7j", help="LLM API key")
    parser.add_argument('--g_base_url',type=str, required=False, default="https://api.chatanywhere.tech/v1", help="base_url")
    parser.add_argument('--g_temperature',type=float, required=False, default=g_temperature, help="temperature parameter for LLM")
    parser.add_argument('--g_top_p',type=float, required=False, default=0.92, help="top_p parameter for LLM")
    parser.add_argument('--g_temperature_step',type=float, required=False, default=0.05, help="temperature step increment for variants")
    parser.add_argument('--g_top_p_step',type=float, required=False, default=0.01, help="top_p step increment for variants")
    parser.add_argument('--g_variants_per_problem',type=int, required=False, default=g_variants_per_problem, help="number of generated samples")

    # revision_llm_model
    parser.add_argument('--r_llm_model',type=str, required=False, default=r_llm_model, help="LLM model name, options: [gpt-4o-mini, claude-3-7-sonnet-20250219, deepseek-r1-250528, o4-mini]")
    parser.add_argument('--r_api_key',type=str, required=False, default="sk-sJxxxxxxxxxxxxEQbn4wgZlMlYTUBiNbGYIUWoqCEPAu1Jwdm7j", help="LLM API key")
    parser.add_argument('--r_base_url',type=str, required=False, default="https://api.chatanywhere.tech/v1", help="base_url")
    parser.add_argument('--r_temperature',type=float, required=False, default=0.5, help="temperature parameter for LLM")
    parser.add_argument('--max_iterations',type=int, required=False, default=3, help="maximum iterations for single variant")
    parser.add_argument('--convergence_patience',type=int, required=False, default=3, help="number of consecutive iterations with no change required for convergence")


    return parser

async def main_experiment(llm_model): # main experiment 
    # datasets = ["complexor", "industryor_easy", "industryor_medium", "industryor_hard",  "mamo_complex", "nl4opt","nlp4lp", "optmath","optibench"]
    # datasets = [ "industryor_medium"]
    # datasets = [ "industryor_hard"]
    # datasets = [ "industryor_easy"]
    # datasets = [ "industryor_medium", "industryor_hard","industryor_easy"]
    datasets = ["1"] # 1 and 2 are the toy test datasets
    for dataset in datasets: 
        print(f"Now training {dataset} with {llm_model}")
        mode = "main" # IO, CoT, main
        args = get_parser(mode, llm_model[0], llm_model[1], 0.7, 3, dataset).parse_args()
        generator = Generator(args)
        await generator.forward()
        # Only main mode executes revision:
        if mode == "main":
            revision = Revision(args)
            await revision.forward()
        

async def main():
    fix_seed = 2021
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    # print("==========Start doing additional experiment 1 to find the best llm_model==========")
    # best_llm_model = ['o4-mini', 'o4-mini']
    best_llm_model = ['o4-mini', 'o4-mini']
    # ['claude-3-7-sonnet-20250219', 'claude-3-7-sonnet-20250219']
    # ['deepseek-r1-250528', 'deepseek-r1-250528']、
    # ['o4-mini', 'o4-mini']
    print("==========Start doing main experiment==========")
    await main_experiment(best_llm_model)

if __name__ == "__main__":
    asyncio.run(main())

