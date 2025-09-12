from openai import AsyncOpenAI
import os
import json
import subprocess
import time
from tqdm import tqdm
import pandas as pd
import jsonlines
import re
import shutil
import asyncio

class Generator:
    def __init__(self, configs):
        self.configs = configs
        self.train_flag = configs.train_flag
        self.dataset = configs.dataset # dataset name
        self.data_path = os.path.join(configs.data_path, self.dataset + ".jsonl") # dataset path
        self.output_path = os.path.join(configs.output_path, self.dataset) # output path
        # IO and CoT modes generate only one variant, main mode generates multiple variants
        if self.train_flag in ["IO", "CoT"]:
            self.variants_per_problem = 1
        else:
            self.variants_per_problem = configs.g_variants_per_problem # number of generated samples
        self.temperature = configs.g_temperature # LLM temperature parameter
        self.temperature_step = configs.g_temperature_step # LLM temperature increment step
        self.top_p = configs.g_top_p # LLM top_p parameter
        self.top_p_step = configs.g_top_p_step # LLM top_p increment step

        self.llm_model = configs.g_llm_model
        self.api_key = configs.g_api_key
        self.base_url = configs.g_base_url
        self.client = AsyncOpenAI(
            api_key= self.api_key, 
            base_url = self.base_url
        )

        self.cache_path = os.path.join(os.path.join(os.path.join(os.path.join(configs.cache_path, self.dataset), configs.g_llm_model), configs.train_flag), "gen_cache.txt")
        self.cache = []
        os.makedirs(os.path.join(os.path.join(os.path.join(configs.cache_path, self.dataset), configs.g_llm_model), configs.train_flag), exist_ok=True)
        if not os.path.exists(self.cache_path):
            with open(self.cache_path, 'w') as f:
                f.write("")  # initialize as empty list
        try:
            with open(self.cache_path, "r") as f:
                for line in f:
                    self.cache.append(line[:line.find('\n')])
        except Exception:
            self.cache = []

    def count_jsonl_lines(self):
        """Count the number of lines in the dataset JSONL file"""
        with jsonlines.open(self.data_path) as reader:
            return len(list(reader))

    def get_prompt(self, problem_description):
        """Generate prompt template based on problem description"""
        if self.train_flag == "IO":
            prompt_template = """
                You are an expert operations research analyst. Your task is to generate Gurobi code to solve the following optimization problem:

                {problem_description}

                The code should save the final optimal value in a file named 'ref_optimal_value.txt'.

                Generate the complete and runnable gurobipy code that solves the problem. Include the model definition, variables, constraints, objective function, and optimization. Put the code between two '=====' lines, like this:

                =====
                # code starts
                import ...
                ...
                # code ends
                =====

                - Do not generate anything after the second '====='.
            """
        elif self.train_flag == "CoT":
            prompt_template = """
                You are an expert operations research analyst. Your task is to solve the following optimization problem using Gurobi:

                {problem_description}

                Before coding, reason step by step:

                1. Decision variables
                - What values need to be decided? What are their types (integer, binary, continuous)?

                2. Objective function
                - What should be optimized (e.g., minimize cost)? Express it with the variables.

                3. Constraints
                - What limits or rules apply? Formulate them mathematically.

                4. Model overview
                - Briefly describe the model type and assumptions.

                Then, write complete gurobipy code to solve the problem. Include:
                - model definition, variables, constraints, objective, optimization, and save the result to `'ref_optimal_value.txt'`.

                Enclose the code between:

                =====
                # code starts
                import ...
                ...
                # code ends
                =====

                Do not output anything after the second '====='.
            """
        else:
            prompt_template = """
            You are an expert operations research analyst. Your task is to solve the following optimization problem:
                {problem_description}
            Your output should consist of two parts:
            1. First, analyze the problem and explicitly formulate its key modeling components in the Five-Element format:
                - **Sets**: Define index sets such as time, location, product, etc.
                - **Variables**: Define decision variables including their type (continuous/integer/binary).
                - **Parameters**: Define the known input quantities.
                - **Constraints**: Describe all constraints the solution must satisfy.
                - **Objective**: Define the goal of the optimization (e.g., minimize cost, maximize output).
                Put the full Five-Element analysis between two `+++++` lines like this:
                +++++
                Sets:
                Variables:
                Parameters:
                Constraints:
                Objective:
                +++++
            2. Then, based on your structured modeling above, generate complete Python code using Gurobi (gurobipy) to solve the problem. Your code should:
                - Be fully executable;
                - Save the final optimal value in a file named `ref_optimal_value.txt`;
                - Include model definition, variables, constraints, objective, and optimization call.
                Put the code between two `=====` lines like this:
            =====
            import ...
            ...
            =====
            - The code should save the final optimal value in a file named 'ref_optimal_value.txt'.
            - Do not generate anything outside the `+++++` and `=====` blocks.
            - Take a deep breath and think step by step.
            """
        
        prompt = prompt_template.format(problem_description=problem_description)
        return prompt

    def extract_code(self, text):
        ele_ind1 = text.find("+++++")
        ele_ind2 = text.find("+++++", ele_ind1 + 5)

        ele = text[ele_ind1 + 5 : ele_ind2].strip()

        ind1 = text.find("=====")
        ind2 = text.find("=====", ind1 + 5)

        code = text[ind1 + 5 : ind2].strip()
        code = code.replace("```python", "").replace("```", "").strip()

        return ele, code

    def execute_code(self, file_path):
        """Execute code and return output and status"""
        try:
            # Use Python's subprocess module to execute code in a separate process
            result = subprocess.run(
                ["python", file_path], capture_output=True, text=True, check=True
            )
            return result.stdout, "Success"
        except subprocess.CalledProcessError as e:
            return e.stderr, "Error"

    def save_response(self, response, output_dir):
        """Save response to file"""
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(output_dir)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        # Write to file
        with open(output_dir, "w") as f:
            f.write(response)

    async def generate_response(self, prompt, variant_num):
        """Generate response using OpenAI API through ChatAnywhere proxy"""
        # IO and CoT modes use fixed temperature, main mode increases temperature with variant number
        if self.train_flag in ["IO", "CoT"]:
            temperature = self.temperature
            top_p = self.top_p
        else:
            temperature = self.temperature + (variant_num * self.temperature_step)  # increase temperature with variant number
            top_p = self.top_p + (variant_num * self.top_p_step)      # slightly adjust top_p
        temperature = min(temperature, 1.3)
        top_p = min(top_p, 0.98)
        
        max_retries = 5
        retries = 0
        while retries < max_retries:
            try:
                start_time = time.time()
                response = await self.client.chat.completions.create(
                    model = self.llm_model,
                    messages = [
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    top_p=top_p
                )
                response_text = response.choices[0].message.content
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Generated variant {variant_num} with temp={temperature:.2f}, top_p={top_p:.2f} in {elapsed_time:.2f}s")
                return response_text, elapsed_time
            except Exception as e:
                print(f"Error occurred: {e}, retrying... ({retries + 1}/{max_retries})")
                retries += 1
                await asyncio.sleep(1)

    async def forward(self):
        """Evaluation function, generate multiple variants and record results"""
        # Main output DataFrame to store all results
        df_output = pd.DataFrame(columns=[
            'id', 'variant_id', 'question', 
            'ground_truth', 'code', 'code_excute_result', 'status'
        ])
    
        total_count = self.count_jsonl_lines()
        print('total_count:', total_count)

        # Ensure base output directory exists
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.output_path = os.path.join(os.path.join(self.output_path, self.llm_model), self.train_flag)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        cache_file = open(self.cache_path, "r+") # open cache file
        cache_file.seek(0,2) # Seek to the end of the file.
        with tqdm(total=total_count, desc="Processing problems") as pbar:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # Parse JSON object
                    data = json.loads(line)
                    id_i = str(data['id'])
                    problem_description = data['question']
                    ground_truth = data['ground_truth']

                    # Cache check
                    if id_i in self.cache:
                        pbar.set_description(f"Skipping completed problem {id_i}")
                        pbar.update(1)  # Update progress for completed problems
                        continue
                    
                    pbar.set_description(f"Processing problem {id_i}")
                    
                    # Create problem-specific directory
                    problem_dir = os.path.join(self.output_path, f"problem_{id_i}")
                    if not os.path.exists(problem_dir):
                        os.makedirs(problem_dir)
                    
                    # Save problem description to problem directory
                    with open(os.path.join(problem_dir, "problem_description.txt"), "w") as f:
                        f.write(problem_description)
                        
                    # Save ground truth to problem directory
                    with open(os.path.join(problem_dir, "ground_truth.txt"), "w") as f:
                        f.write(str(ground_truth))  # convert to string
                    
                    # Create prompt for this problem
                    prompt = self.get_prompt(problem_description)
                    print(f"\nGenerating {self.variants_per_problem} variants for problem {id_i}")
                    
                    # Create variants directory within problem directory
                    variants_dir = os.path.join(problem_dir, "variants")
                    if not os.path.exists(variants_dir):
                        os.makedirs(variants_dir)
                    
                    # Track variants for this problem
                    problem_variants = []
                    
                    # Generate multiple variants for this problem
                    for variant_num in range(self.variants_per_problem):
                        variant_id = f"{id_i}_{variant_num}"
                        
                        # Create specific directory for this variant
                        variant_dir = os.path.join(variants_dir, f"variant_{variant_num}")
                        if not os.path.exists(variant_dir):
                            os.makedirs(variant_dir)
                        
                        print(f"Generating variant {variant_num+1}/{self.variants_per_problem}")
                        
                        # Use OpenAI API to generate different solutions
                        result = await self.generate_response(
                            prompt, 
                            variant_num=variant_num
                        )
                        if result is None:
                            print(f"Failed to generate response for variant {variant_num} of problem {id_i}")
                            continue
                        response, elapsed_time = result
                        
                        # Save complete response to variant directory
                        response_path = os.path.join(variant_dir, "response.txt")
                        self.save_response(response, response_path)
                        
                        # Extract code
                        ele, code = self.extract_code(response)
                        print(f'Processing variant {variant_id}')
                        
                        # Save code to files in variant directory
                        ele_filename = "5element_description.txt"
                        code_filename = "solution.py"
                        ele_file_path = os.path.join(variant_dir, ele_filename)
                        code_file_path = os.path.join(variant_dir, code_filename)
                        with open(ele_file_path, "w") as f:
                            f.write(ele)
                        with open(code_file_path, "w") as f:
                            f.write(code)

                        # Execute code
                        output, status = self.execute_code(code_file_path)

                        # Save execution output
                        output_path = os.path.join(variant_dir, "execution_output.txt")
                        with open(output_path, "w") as f:
                            f.write(f"Status: {status}\n\n")
                            f.write(output)

                        # Save error output (if any)
                        if status == "Error":
                            error_filename = "error.txt"
                            with open(os.path.join(variant_dir, error_filename), "w") as f:
                                f.write(output)

                        if status == "Success":
                            print(f'Variant {variant_id}: Code executed successfully')
                            ref_optimal_path = "ref_optimal_value.txt"
                            if os.path.exists(ref_optimal_path):
                                shutil.copy(ref_optimal_path, os.path.join(variant_dir, ref_optimal_path))
                                os.remove(ref_optimal_path)
                        else:
                            print(f"Variant {variant_id}: Error encountered")
                        
                        problem_variants.append({
                            'variant_num': variant_num,
                            'status': status,
                            'path': variant_dir
                        })
                            
                            
                        dict_temp = {
                            'id': id_i,
                            'variant_id': variant_id,
                            'question': problem_description,
                            'ground_truth': ground_truth,
                            'code': code,
                            'code_excute_result': output,
                            'status': status
                        }
                        df_temp = pd.DataFrame([dict_temp])
                        df_output = pd.concat([df_output, df_temp], ignore_index=True)
                    
                    self.cache.append(str(id_i)) # save the id
                    cache_file.write(str(id_i) + "\n") # append the id in cache_file

                    # Create summary file for all variants of this problem
                    summary_path = os.path.join(problem_dir, "variants_summary.csv")
                    variants_df = pd.DataFrame(problem_variants)
                    variants_df.to_csv(summary_path, index=False)
                    
                    # Save intermediate results after processing each problem
                    output_name = os.path.join(self.output_path, 'all_results.csv')
                    df_output.to_csv(output_name, index=False)
                    
                    # Update progress bar
                    pbar.update(1)
                
        # Finally save main DataFrame
        df_output.to_csv(os.path.join(self.output_path, 'all_results.csv'), index=False)

        cache_file.close()

        # Calculate and save success statistics
        success_stats = df_output.groupby(['id']).agg({
            'status': lambda x: (x == 'Success').sum(),  # number of successful variants
            'variant_id': 'count'  # total number of variants
        }).reset_index()
        success_stats.columns = ['id', 'successful_variants', 'total_variants']
        success_stats['success_rate'] = success_stats['successful_variants'] / success_stats['total_variants']
        success_stats.to_csv(os.path.join(self.output_path, 'success_statistics.csv'), index=False)
        
        print(f"\nVariant generation complete. Results saved to {self.output_path}")


class Revision:
    def __init__(self, configs):
        self.train_flag = configs.train_flag
        self.path = os.path.join(os.path.join(os.path.join(configs.output_path, configs.dataset), configs.g_llm_model), self.train_flag)
            
        self.max_iterations = configs.max_iterations # maximum iterations for variant
        self.temperature = configs.r_temperature
        self.llm_model = configs.r_llm_model
        self.api_key=configs.r_api_key
        self.base_url=configs.r_base_url
        self.client = AsyncOpenAI(
            api_key = self.api_key, 
            base_url = self.base_url
        )

        self.cache_path = os.path.join(os.path.join(os.path.join(os.path.join(configs.cache_path, configs.dataset), configs.g_llm_model), configs.train_flag), "ref_cache.txt")
        self.cache = []
        os.makedirs(os.path.join(os.path.join(os.path.join(configs.cache_path, configs.dataset), configs.g_llm_model), configs.train_flag), exist_ok=True)
        if not os.path.exists(self.cache_path):
            with open(self.cache_path, 'w') as f:
                f.write("")  # initialize as empty list
        try:
            with open(self.cache_path, "r") as f:
                for line in f:
                    self.cache.append(line[:line.find('\n')])
        except Exception:
            self.cache = []

    def extract_code(self, text):
        ind1 = text.find("=====")
        ind2 = text.find("=====", ind1 + 5)
        code = text[ind1 + 5 : ind2].strip()
        code = code.replace("```python", "").replace("```", "").strip()
        return code
    
    def execute_code(self, file_path):
        """Execute code and return output and status"""
        try:
            # Use Python's subprocess module to execute code in a separate process
            result = subprocess.run(
                ["python", file_path], capture_output=True, text=True, check=True, timeout=300
            )
            return result.stdout, "Success"
        except subprocess.CalledProcessError as e:
            return e.stderr, "Error"
        except subprocess.TimeoutExpired:
            return "Code execution timed out after 5 minutes.", "Timeout"

    def read_optimal_value(self, file_path):
        """Read optimal value from file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    return content  # return original content without conversion
            return None
        except Exception as e:
            print(f"Error reading optimal value: {e}")
            return None
    
    def get_problem_description(self, problem_dir):
        """Read problem description from file"""
        problem_file = os.path.join(problem_dir, "problem_description.txt")
        if os.path.exists(problem_file):
            with open(problem_file, 'r') as f:
                return f.read()
        return None
    
    def get_ground_truth(self, problem_dir):
        """Read ground truth from file"""
        ground_truth_file = os.path.join(problem_dir, "ground_truth.txt")
        if os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r') as f:
                content = f.read().strip()
                return content  # 返回原始内容，不尝试转换
        return None

    def get_variant_5element(self, variant_path):
        """Get 5-element description of the variant"""
        element_file = os.path.join(variant_path, "5element_description.txt")
        if os.path.exists(element_file):
            with open(element_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def extract_numerical_value(self, text):
        """Extract numerical value from text
        
        Args:
            text (str): Text containing numerical value, e.g., "Optimal Value: 11000.0"
        
        Returns:
            str or None: Extracted numerical value string, None if not found
        """
        if text is None:
            return None
        
        # Use regular expressions to extract numerical values
        import re
        # Find floating point numbers or integers in text
        pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(pattern, text)
        
        if matches:
            # Return the first numerical value found
            return matches[0]
        
        return None

    def compare_with_ground_truth(self, value, ground_truth):
        """Compare a value with ground truth, without extracting numerical value for ground_truth"""
        if value is None or ground_truth is None:
            return False
        
        # Only extract numerical value for computed value, keep ground_truth as is
        extracted_value = self.extract_numerical_value(value)
        
        # If numerical value can be extracted, perform numerical comparison
        if extracted_value:
            try:
                value_float = float(extracted_value)
                ground_truth_float = float(ground_truth)  # directly try to convert ground_truth
                
                # For floating point numbers, use approximate equality
                return abs(value_float - ground_truth_float) < 1e-6 * max(1, abs(ground_truth_float))
            except (ValueError, TypeError):
                # If ground_truth cannot be directly converted to float, fall back to text comparison
                pass
        
        # If numerical extraction fails or conversion fails, fall back to text comparison
        # First try to clean text (remove all whitespace and punctuation)
        def clean_text(text):
            if text is None:
                return ""
            return re.sub(r'[\s\.,;:!?]+', '', text.lower())
        
        cleaned_value = clean_text(value)
        cleaned_ground_truth = clean_text(ground_truth)
        
        return cleaned_value == cleaned_ground_truth

    def get_variant_data(self, variant_path):
        """Get all data for a variant"""
        variant_data = {"path": variant_path}
        
        # Read code
        code_path = os.path.join(variant_path, "solution.py")
        if os.path.exists(code_path):
            with open(code_path, 'r', encoding='utf-8') as f:
                variant_data["code"] = f.read()
        
        # Read 5-element description
        element_description = self.get_variant_5element(variant_path)
        if element_description:
            variant_data["5element_description"] = element_description
        
        # Check execution output
        output_path = os.path.join(variant_path, "execution_output.txt")
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                output_content = f.read()
                variant_data["execution_output"] = output_content
                variant_data["status"] = "Success" if "Status: Success" in output_content else "Error"
        
        # Check error file
        error_path = os.path.join(variant_path, "error.txt")
        if os.path.exists(error_path):
            with open(error_path, 'r', encoding='utf-8') as f:
                variant_data["error"] = f.read()
        
        # Check optimal value
        optimal_path = os.path.join(variant_path, "ref_optimal_value.txt")
        if os.path.exists(optimal_path):
            variant_data["optimal_value"] = self.read_optimal_value(optimal_path)
        
        return variant_data

    def generate_evaluation_prompt(self, variant_data, problem_description, iteration_number):
        """Generate multi-dimensional evaluation prompt (without ground_truth)"""
        prompt = f"""You are an expert operations research analyst. I need your help evaluating and diagnosing issues in an optimization problem solution. Please perform a systematic Chain-of-Thought evaluation across three key dimensions.

    ## Problem Description:
    {problem_description}

    ## Current Iteration: {iteration_number}

    ## Variant Information:
    """
        
        # Add 5-element description (if available)
        if "5element_description" in variant_data:
            prompt += f"\n### 5-Element Problem Analysis:\n{variant_data['5element_description']}\n"
        
        prompt += f"\n### Current Solution Code:\n```python\n{variant_data.get('code', 'No code available')}\n```\n"
        
        # Add execution results
        if variant_data.get("status") == "Success":
            prompt += f"\n### Execution Result: SUCCESS\n"
            if "optimal_value" in variant_data:
                prompt += f"Computed optimal value: {variant_data['optimal_value']}\n"
            if "execution_output" in variant_data:
                prompt += f"Full output:\n{variant_data['execution_output']}\n"
        else:
            prompt += f"\n### Execution Result: FAILED\n"
            prompt += f"Error details: {variant_data.get('error', 'No error information available')}\n"
        
        # Multi-dimensional evaluation instructions
        prompt += """
    ## Please perform systematic evaluation across THREE dimensions:

    ### 🔍 Dimension 1: Definition Consistency
    Analyze whether the five element aligns with the problem description:
    - Are all variables properly defined and named according to the problem context?
    - Do the parameter values match what's described in the problem?
    - Are all problem entities (e.g., facilities, customers, resources) correctly represented?
    - Is the problem scope and scale correctly captured?

    ### 🔍 Dimension 2: Structural Soundness  
    Evaluate the mathematical model structure:
    - Is the objective function correctly formulated (minimization vs maximization)?
    - Are all necessary constraints included?
    - Are there any missing or redundant constraints?
    - Is the model type appropriate (linear, integer, mixed-integer)?
    - Are variable types (continuous, binary, integer) correctly specified?

    ### 🔍 Dimension 3: Numerical Validity
    Assess the numerical aspects and execution:
    - If successful: Does the optimal value seem reasonable given the problem context?
    - If failed: What is the root cause of the execution error?
    - Are there any obvious numerical issues (infeasibility, unboundedness)?
    - Are the constraint right-hand-sides and coefficients reasonable?

    ## Required Output Format:

    **DIMENSION 1 - Definition Consistency:**
    Diagnosis: [Your analysis]
    Issues Found: [List specific issues or "None"]
    Confidence: [High/Medium/Low]

    **DIMENSION 2 - Structural Soundness:**  
    Diagnosis: [Your analysis]
    Issues Found: [List specific issues or "None"]
    Confidence: [High/Medium/Low]

    **DIMENSION 3 - Numerical Validity:**
    Diagnosis: [Your analysis] 
    Issues Found: [List specific issues or "None"]
    Confidence: [High/Medium/Low]

    **OVERALL ASSESSMENT:**
    Primary Issues: [Rank the most critical issues to fix]
    Recommended Actions: [Specific actionable recommendations]

    Please provide thorough, specific analysis for each dimension. Focus on identifying concrete, fixable issues rather than general observations.
    """
        
        return prompt

    def generate_refinement_prompt(self, evaluation_result, variant_data, problem_description, iteration_number):
        """Generate improvement prompt based on evaluation results (without ground_truth)"""
        prompt = f"""You are an expert operations research analyst. Based on the detailed diagnostic evaluation below, please generate an improved Gurobi solution for this optimization problem.

    ## Problem Description:
    {problem_description}

    ## Current Iteration: {iteration_number}

    ## Diagnostic Evaluation Results:
    {evaluation_result}

    ## Current 5element to Improve:
    {variant_data.get('5element_description', 'No 5element_description')}

    ## Current Solution to Improve:
    ```python
    {variant_data.get('code', 'No code available')}
    ```

    ## Instructions for Improvement:

    Based on the diagnostic evaluation above, please generate a completely new and improved Gurobi code that addresses ALL identified issues. 

    **Requirements:**
    1. Address each issue identified in the diagnostic evaluation
    2. Ensure the code can run successfully without errors
    3. Save the final optimal value to 'ref_optimal_value.txt' (value only, no extra text)
    4. Use proper Gurobi syntax and best practices
    5. Include appropriate error handling
    6. Add clear comments explaining key modeling decisions

    **Code Generation Format:**
    First, provide a brief explanation of the key improvements you're making, then provide the complete code between ===== markers:

    Key Improvements:
    [Explain the main changes you're making to address the identified issues]

    =====
    import gurobipy as gp
    from gurobipy import GRB

    # Your improved code here
    # Make sure to save only the optimal value to 'ref_optimal_value.txt'
    =====

    Please ensure the generated code is complete, executable, and addresses all the issues identified in the diagnostic evaluation.
    """
        
        return prompt

    async def evaluate_variant(self, variant_data, problem_description, iteration_number):
        """Perform multi-dimensional evaluation on variant"""
        evaluation_prompt = self.generate_evaluation_prompt(variant_data, problem_description, iteration_number)
        
        response = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "user", "content": evaluation_prompt},
            ],
            temperature=self.temperature  # low temperature ensures consistent evaluation
        )

        evaluation_result = response.choices[0].message.content
        
        return evaluation_result, evaluation_prompt

    async def generate_refined_code(self, evaluation_result, variant_data, problem_description, iteration_number):
        """Generate improved code based on evaluation results"""
        refinement_prompt = self.generate_refinement_prompt(evaluation_result, variant_data, problem_description, iteration_number)
        
        response = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "user", "content": refinement_prompt},
            ],
            temperature=self.temperature
        )

        response_text = response.choices[0].message.content
        
        # Extract code from response
        code = self.extract_code(response_text)
        
        return code, response_text, refinement_prompt

    def check_correctness_internal(self, optimal_value, ground_truth):
        """Internal correctness check (not exposed to LLM)"""
        return self.compare_with_ground_truth(optimal_value, ground_truth)

    async def refine_variant(self, base_variant_path, problem_description, ground_truth, max_iterations, variant_name):
        """Improve a variant through multiple iterations"""
        
        # Get initial variant data
        variant_data = self.get_variant_data(base_variant_path)
        
        # Create an improvement directory for this variant
        refinement_dir = os.path.join(os.path.dirname(base_variant_path), f"{variant_name}_refinements")
        if not os.path.exists(refinement_dir):
            os.makedirs(refinement_dir)
        
        # Track latest iteration data for continued improvement
        current_variant_data = variant_data
        
        # Start improvement iterations
        for iteration in range(1, max_iterations + 1):
            print(f"Starting {variant_name} refinement iteration {iteration}/{max_iterations}")
            
            # Create iteration directory
            iter_dir = os.path.join(refinement_dir, f"iteration_{iteration}")
            if not os.path.exists(iter_dir):
                os.makedirs(iter_dir)
            
            # Step 1: Multi-dimensional evaluation (without ground_truth)
            print(f"  Step 1: Evaluating variant {variant_name} iteration {iteration}")
            evaluation_result = await self.evaluate_variant(
                current_variant_data, problem_description, iteration
            )
            if evaluation_result is None:
                print(f"Failed to evaluate variant {variant_name} iteration {iteration}")
                continue
            else:
                evaluation_result, evaluation_prompt = evaluation_result
            
            # Save evaluation prompt and results
            with open(os.path.join(iter_dir, "evaluation_prompt.txt"), "w", encoding='utf-8') as f:
                f.write(evaluation_prompt)
            with open(os.path.join(iter_dir, "evaluation_result.txt"), "w", encoding='utf-8') as f:
                f.write(evaluation_result if evaluation_result is not None else "")
            
            # Step 2: Generate improved code (without ground_truth)
            print(f"  Step 2: Generating refined code for {variant_name} iteration {iteration}")
            code, response, refinement_prompt = await self.generate_refined_code(
                evaluation_result, current_variant_data, problem_description, iteration
            )
            
            # Save improvement prompt and complete response
            with open(os.path.join(iter_dir, "refinement_prompt.txt"), "w", encoding='utf-8') as f:
                f.write(refinement_prompt)
            with open(os.path.join(iter_dir, "refinement_response.txt"), "w", encoding='utf-8') as f:
                f.write(response if response is not None else "")
            
            # Save code
            code_path = os.path.join(iter_dir, "solution.py")
            with open(code_path, "w", encoding='utf-8') as f:
                f.write(code)
            
            # 步骤3: 执行代码
            print(f"  Step 3: Executing refined code for {variant_name} iteration {iteration}")
            output, status = self.execute_code(code_path)
            
            # Save execution output
            with open(os.path.join(iter_dir, "execution_output.txt"), "w", encoding='utf-8') as f:
                f.write(f"Status: {status}\n\n")
                f.write(output)
            
            # Save error output (if any)
            if status != "Success":
                with open(os.path.join(iter_dir, "error.txt"), "w", encoding='utf-8') as f:
                    f.write(output)
                print(f"    ❌ {variant_name} iteration {iteration} failed: Code execution error")
            else:
                print(f"    ✓ {variant_name} iteration {iteration} executed successfully")
                
                # Check if ref_optimal_value.txt file was generated
                ref_optimal_path = "ref_optimal_value.txt"
                if os.path.exists(ref_optimal_path):
                    # Copy file to iteration directory
                    shutil.copy(ref_optimal_path, os.path.join(iter_dir, ref_optimal_path))
                    # Delete temporary file from root directory
                    os.remove(ref_optimal_path)
                    
                    # Read optimal value
                    optimal_value = self.read_optimal_value(ref_optimal_path)
                    
                    # Internal correctness check (use ground_truth but don't expose to LLM)
                    if self.check_correctness_internal(optimal_value, ground_truth):
                        print(f"✅ {variant_name} solved correctly in iteration {iteration}")
                        return {
                            "status": "Solved",
                            "iterations": iteration,
                            "optimal_value": optimal_value,
                            "ground_truth": ground_truth,
                            "path": iter_dir
                        }
                    else:
                        # Print debug information (but don't pass to LLM)
                        extracted_optimal = self.extract_numerical_value(optimal_value)
                        print(f"⚠️ {variant_name} iteration {iteration} produced incorrect result:")
                        print(f"  Raw optimal: {optimal_value}")
                        print(f"  Extracted optimal: {extracted_optimal}")
                        print(f"  Ground truth: {ground_truth}")
                else:
                    print(f"⚠️ {variant_name} iteration {iteration} did not produce ref_optimal_value.txt")
            
            # Update variant data with current iteration results for next iteration
            current_variant_data = self.get_variant_data(iter_dir)
        
        return {
            "status": "Not solved",
            "iterations": max_iterations,
            "ground_truth": ground_truth,
            "path": refinement_dir
        }

    def check_variant_correct(self, variant_path, ground_truth):
        """Check if a variant produces correct results"""
        optimal_path = os.path.join(variant_path, "ref_optimal_value.txt")
        if os.path.exists(optimal_path):
            optimal_value = self.read_optimal_value(optimal_path)
            return self.compare_with_ground_truth(optimal_value, ground_truth)
        return False

    async def forward(self):
        """Improve solutions for all problems in the base directory"""
        # Find all problem directories
        problem_dirs = [os.path.join(self.path, d) for d in os.listdir(self.path) 
                    if os.path.isdir(os.path.join(self.path, d)) and d.startswith("problem_")]
        
        results = []
        
        cache_file = open(self.cache_path, "r+") # open cache file
        cache_file.seek(0,2) # Seek to the end of the file.
        with tqdm(total=len(problem_dirs), desc="Processing problems for revision") as pbar:
            for problem_dir in problem_dirs:
                problem_id = os.path.basename(problem_dir).replace("problem_", "")
                print(f"\nProcessing problem {problem_id}")

                # Cache check
                if problem_id in self.cache:
                    pbar.set_description(f"Skipping completed problem {problem_id}")
                    pbar.update(1)
                    continue
                
                pbar.set_description(f"Processing problem {problem_id} for revision")
                
                # Create an improvement directory
                refinement_dir = os.path.join(problem_dir, "refinements")
                if not os.path.exists(refinement_dir):
                    os.makedirs(refinement_dir)
                
                # Get problem description and ground truth
                problem_description = self.get_problem_description(problem_dir)
                ground_truth = self.get_ground_truth(problem_dir)
                
                if not problem_description or ground_truth is None:
                    print(f"Missing problem description or ground truth for problem {problem_id}")
                    pbar.update(1)
                    continue
                
                # Check if any original variants have already produced correct results
                variants_dir = os.path.join(problem_dir, "variants")
                if not os.path.exists(variants_dir):
                    print(f"No variants directory found for problem {problem_id}")
                    pbar.update(1)
                    continue
                
                variant_dirs = [os.path.join(variants_dir, d) for d in sorted(os.listdir(variants_dir))
                    if (os.path.isdir(os.path.join(variants_dir, d)) and "refinements" not in d)]
                
                solved = False
                for variant_dir in variant_dirs:
                    variant_name = os.path.basename(variant_dir)
                    if self.check_variant_correct(variant_dir, ground_truth):
                        print(f"✅ Problem {problem_id}, {variant_name} already solved correctly in first round")
                        solved = True
                        results.append({
                        "problem_id": problem_id,
                        "variant": variant_name,
                        "status": "Solved in first round",
                        "iterations": 0,
                            "ground_truth": ground_truth
                        })
                        break
                
                if solved:
                    self.cache.append(problem_id)
                    pbar.update(1)
                    continue
                
                # If no original variants produced correct results, improve each variant
                print(f"No original variants solved problem {problem_id}. Refining all variants...")
                
                # Track results for this problem
                problem_results = []
                
                # Process each variant independently
                for variant_dir in variant_dirs:
                    variant_name = os.path.basename(variant_dir)
                    print(f"\nRefining {variant_name} for problem {problem_id}")
                    
                    # Improve this variant (pass ground_truth for internal evaluation, but don't pass to LLM)
                    refinement_result = await self.refine_variant(
                        variant_dir, 
                        problem_description, 
                        ground_truth,  # for internal correctness check
                        self.max_iterations,
                        variant_name
                    )
                    
                    # Add problem and variant information
                    refinement_result["problem_id"] = problem_id
                    refinement_result["variant"] = variant_name
                    
                    # Add to results
                    problem_results.append(refinement_result)
                    results.append(refinement_result)
                    
                    # If this variant is solved, we can terminate early
                    if refinement_result["status"] == "Solved":
                        print(f"✅ Problem {problem_id} solved by refining {variant_name}")
                        solved = True
                        break
            
                self.cache.append(problem_id)
                cache_file.write(problem_id + "\n") # append the id in cache_file

                # Create a summary file for improvements to this problem
                summary_path = os.path.join(refinement_dir, "refinement_summary.csv")
                problem_df = pd.DataFrame(problem_results)
                problem_df.to_csv(summary_path, index=False)
                
                if not solved:
                    print(f"❌ Problem {problem_id} not solved after refining all variants")
                
                # Update progress bar
                pbar.update(1)
        
        cache_file.close()
        
        # Compile and save summary results
        results_df = pd.DataFrame(results)
        
        # If only one problem, save results to that problem's refinements directory
        # If multiple problems, save to experiment root directory
        if len(problem_dirs) == 1:
            problem_refinement_dir = os.path.join(problem_dirs[0], "refinements")
            if not os.path.exists(problem_refinement_dir):
                os.makedirs(problem_refinement_dir)
            results_df.to_csv(os.path.join(problem_refinement_dir, "refinement_results.csv"), index=False)
        else:
            results_df.to_csv(os.path.join(self.path, "refinement_results.csv"), index=False)
        
        # Print summary information
        print("\nRefinement Summary:")
        print(f"Total problems processed: {len(set(results_df['problem_id']))}")
        print(f"Problems solved in first round: {len(results_df[results_df['status'] == 'Solved in first round'])}")
        print(f"Problems solved in refinement: {len(results_df[results_df['status'] == 'Solved'])}")
        problems_not_solved = set(results_df['problem_id']) - set(results_df[results_df['status'].isin(['Solved', 'Solved in first round'])]['problem_id'])
        print(f"Problems not solved: {len(problems_not_solved)}")
        return results_df


class calculate_acc:
    def __init__(self, configs):
        self.train_flag = configs.train_flag
        self.llm_model_path = configs.g_llm_model
        self.path = os.path.join(os.path.join(os.path.join(configs.output_path, configs.dataset), configs.g_llm_model), self.train_flag)
    
    def read_optimal_value(self, file_path):
        """Read optimal value from file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    return content
            return None
        except Exception as e:
            print(f"Error reading optimal value: {e}")
            return None

    def get_ground_truth(self, problem_dir):
        """Read ground truth from file"""
        ground_truth_file = os.path.join(problem_dir, "ground_truth.txt")
        if os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r') as f:
                content = f.read().strip()
                return content
        return None

    def extract_numerical_value(self, text):
        """Extract numerical value from text
        
        Args:
            text (str): Text containing numerical value, e.g., "Optimal Value: 11000.0"
        
        Returns:
            str or None: Extracted numerical value string, None if not found
        """
        if text is None:
            return None
        
        # Use regular expressions to extract numerical values
        pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(pattern, text)
        
        if matches:
            # Return the first numerical value found
            return matches[0]
        
        return None

    def compare_with_ground_truth(self, value, ground_truth):
        """Compare a value with ground truth"""
        if value is None or ground_truth is None:
            return False
        
        # Extract numerical values
        extracted_value = self.extract_numerical_value(value)
        extracted_ground_truth = ground_truth
        
        # If both can extract numerical values, perform numerical comparison
        if extracted_value and extracted_ground_truth:
            try:
                value_float = float(extracted_value)
                ground_truth_float = float(extracted_ground_truth)
                
                # For floating point numbers, use approximate equality
                is_equal = abs(value_float - ground_truth_float) < 1e-6 * max(1, abs(ground_truth_float))
                
                if is_equal:
                    print(f"✓ Numerical comparison: {value_float} ≈ {ground_truth_float}")
                else:
                    print(f"✗ Numerical comparison: {value_float} ≠ {ground_truth_float}")
                    
                return is_equal
            except (ValueError, TypeError):
                print(f"⚠️ Unable to convert extracted values to float: {extracted_value}, {extracted_ground_truth}")
        
        # If numerical extraction fails or conversion fails, fall back to text comparison
        # First try to clean text (remove all whitespace and punctuation)
        def clean_text(text):
            if text is None:
                return ""
            return re.sub(r'[\s\.,;:!?]+', '', text.lower())
        
        cleaned_value = clean_text(value)
        cleaned_ground_truth = clean_text(ground_truth)
        
        is_equal = cleaned_value == cleaned_ground_truth
        if is_equal:
            print(f"✓ Text comparison: cleaned text is identical")
        else:
            print(f"✗ Text comparison: cleaned text is different")
            print(f"  Value 1 (cleaned): {cleaned_value}")
            print(f"  Value 2 (cleaned): {cleaned_ground_truth}")
        
        return is_equal

    def check_variant_correct(self, variant_path, ground_truth):
        """Check if a variant produces correct results"""
        optimal_path = os.path.join(variant_path, "ref_optimal_value.txt")
        if os.path.exists(optimal_path):
            optimal_value = self.read_optimal_value(optimal_path)
            print(f"Comparing values: \n  Computed value: {optimal_value}\n  Ground truth: {ground_truth}")
            
            # Extract numerical values for information display
            extracted_optimal = self.extract_numerical_value(optimal_value)
            extracted_ground_truth = ground_truth
            if extracted_optimal and extracted_ground_truth:
                print(f"Extracted values: \n  Computed value: {extracted_optimal}\n  Ground truth: {extracted_ground_truth}")
            
            return self.compare_with_ground_truth(optimal_value, ground_truth)
        return False

    def forward(self):
        """Calculate success rate for all problems"""
        # Find all problem directories
        problem_dirs = [os.path.join(self.path, d) for d in os.listdir(self.path) 
                    if os.path.isdir(os.path.join(self.path, d)) and d.startswith("problem_")]
        
        results = []
        
        for problem_dir in tqdm(problem_dirs, desc="Calculating problem success rate"):
            problem_id = os.path.basename(problem_dir).replace("problem_", "")
            print(f"\nEvaluating problem {problem_id}")
            
            # Get ground truth
            ground_truth = self.get_ground_truth(problem_dir)
            
            if ground_truth is None:
                print(f"Problem {problem_id} missing ground truth")
                results.append({
                    "problem_id": problem_id,
                    "solved": False,
                    "reason": "missing ground truth"
                })
                continue
            
            # Check variants directory
            variants_dir = os.path.join(problem_dir, "variants")
            if not os.path.exists(variants_dir):
                print(f"Problem {problem_id} variants directory not found")
                results.append({
                    "problem_id": problem_id,
                    "solved": False,
                    "reason": "no variants directory"
                })
                continue
            
            variant_dirs = [os.path.join(variants_dir, d) for d in sorted(os.listdir(variants_dir))
                if os.path.isdir(os.path.join(variants_dir, d))]
            
            # A problem is considered solved if any variant produces the correct answer
            problem_solved = False
            
            # First check original variants
            for variant_dir in variant_dirs:
                variant_name = os.path.basename(variant_dir)
                print(f"Checking variant {variant_name}")
                if self.check_variant_correct(variant_dir, ground_truth):
                    print(f"✅ Problem {problem_id}, {variant_name} correctly solved in original round")
                    problem_solved = True
                    break
            
            # If original variants are not solved, check improvement iterations
            if not problem_solved:
                # Check if there is an improvement directory
                refinements_dir = os.path.join(problem_dir, "refinements")
                if os.path.exists(refinements_dir):
                    # Check improvement directory for each variant
                    variant_refinement_dirs = [d for d in os.listdir(refinements_dir) 
                                            if os.path.isdir(os.path.join(refinements_dir, d)) and d.endswith("_refinements")]
                    
                    for variant_ref_name in variant_refinement_dirs:
                        variant_ref_path = os.path.join(refinements_dir, variant_ref_name)
                        print(f"Checking variant improvement {variant_ref_name}")
                        
                        # Check each iteration of this variant
                        iteration_dirs = [os.path.join(variant_ref_path, d) for d in sorted(os.listdir(variant_ref_path))
                                        if os.path.isdir(os.path.join(variant_ref_path, d)) and d.startswith("iteration_")]
                        
                        for iter_dir in iteration_dirs:
                            iter_name = os.path.basename(iter_dir)
                            print(f"Checking iteration {iter_name}")
                            if self.check_variant_correct(iter_dir, ground_truth):
                                variant_base_name = variant_ref_name.replace("_refinements", "")
                                print(f"✅ Problem {problem_id} correctly solved in {iter_name} of {variant_base_name}")
                                problem_solved = True
                                break
                        
                        if problem_solved:
                            break
            
            # Record results
            results.append({
                "problem_id": problem_id,
                "solved": problem_solved,
                "ground_truth": ground_truth
            })
            
            if problem_solved:
                print(f"✅ Problem {problem_id} solved")
            else:
                print(f"❌ Problem {problem_id} not solved")
        
        # Create summary dataframe
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(os.path.join(self.path, "problem_success_rate.csv"), index=False)
        
        # Generate summary statistics
        total_problems = len(results_df)
        solved_problems = len(results_df[results_df['solved'] == True])
        success_rate = solved_problems / total_problems * 100 if total_problems > 0 else 0
        
        print("\nSuccess Rate Statistics:")
        print(f"Total problems: {total_problems}")
        print(f"Solved problems: {solved_problems}")
        print(f"Success rate: {success_rate:.2f}%")
        
        # Generate detailed report
        report_path = os.path.join(self.path, "success_rate_report.txt")
        with open(report_path, "w") as f:
            f.write("# Optimization Problem Success Rate Report\n\n")
            f.write(f"Total problems: {total_problems}\n")
            f.write(f"Solved problems: {solved_problems}\n")
            f.write(f"Success rate: {success_rate:.2f}%\n\n")
            
            f.write("## Solved Problems List\n\n")
            for _, row in results_df[results_df['solved'] == True].iterrows():
                f.write(f"- Problem {row['problem_id']}\n")
            
            f.write("\n## Unsolved Problems List\n\n")
            for _, row in results_df[results_df['solved'] == False].iterrows():
                f.write(f"- Problem {row['problem_id']}\n")
        
        return {
            "total_problems": total_problems,
            "solved_problems": solved_problems,
            "success_rate": success_rate
        }
