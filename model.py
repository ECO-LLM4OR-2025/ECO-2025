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

    def parse_objective_direction(self, five_element_text):
        """Parse objective direction (maximize/minimize)"""
        if not five_element_text:
            return None
        
        text_lower = five_element_text.lower()
        
        # Find Objective section
        obj_start = text_lower.find("objective:")
        if obj_start == -1:
            obj_start = text_lower.find("objective")
        
        if obj_start != -1:
            # Extract content from Objective section
            obj_section = text_lower[obj_start:obj_start + 200]  # Take 200 characters
            
            if "maximize" in obj_section or "max" in obj_section:
                return "maximize"
            elif "minimize" in obj_section or "min" in obj_section:
                return "minimize"
        
        return None

    def extract_numerical_value(self, text):
        """Extract numerical value from text"""
        if text is None:
            return None
        
        # Use regular expression to extract numerical value
        pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(pattern, text)
        
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                return None
        
        return None

    def check_convergence(self, prev_value, curr_value, prev_code, curr_code, tolerance=1e-6):
        """Check if converged (no significant change in value and code)"""
        # Check numerical convergence
        if prev_value is not None and curr_value is not None:
            if abs(prev_value - curr_value) < tolerance:
                return True
        
        # Check code convergence (simple string comparison)
        if prev_code and curr_code:
            if prev_code.strip() == curr_code.strip():
                return True
        
        return False

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
            'ground_truth', 'code', 'code_excute_result', 'status', 'objective_direction', 'optimal_value'
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
                    objective_directions = []  # Collect objective directions from all variants
                    
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
                        
                        # Extract code and five-element description
                        ele, code = self.extract_code(response)
                        print(f'Processing variant {variant_id}')
                        
                        # Parse objective direction
                        objective_direction = self.parse_objective_direction(ele)
                        if objective_direction:
                            objective_directions.append(objective_direction)
                        
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
                        optimal_value = None

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
                                # Read optimal value
                                with open(ref_optimal_path, 'r') as f:
                                    optimal_value = f.read().strip()
                                os.remove(ref_optimal_path)
                        else:
                            print(f"Variant {variant_id}: Error encountered")
                        
                        problem_variants.append({
                            'variant_num': variant_num,
                            'status': status,
                            'path': variant_dir,
                            'objective_direction': objective_direction,
                            'optimal_value': optimal_value
                        })
                            
                        dict_temp = {
                            'id': id_i,
                            'variant_id': variant_id,
                            'question': problem_description,
                            'ground_truth': ground_truth,
                            'code': code,
                            'code_excute_result': output,
                            'status': status,
                            'objective_direction': objective_direction,
                            'optimal_value': optimal_value
                        }
                        df_temp = pd.DataFrame([dict_temp])
                        df_output = pd.concat([df_output, df_temp], ignore_index=True)
                    
                    # Determine consensus objective direction
                    consensus_direction = None
                    if objective_directions:
                        # If all variants have the same objective direction, use it as consensus
                        if len(set(objective_directions)) == 1:
                            consensus_direction = objective_directions[0]
                        else:
                            # If inconsistent, choose the most frequent one
                            from collections import Counter
                            consensus_direction = Counter(objective_directions).most_common(1)[0][0]
                    
                    # Save objective direction to problem directory
                    with open(os.path.join(problem_dir, "objective_direction.txt"), "w") as f:
                        f.write(consensus_direction or "unknown")
                    
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
        
        print(f"\nVariant generation complete. Results saved to {self.output_path}")


class Revision:
    def __init__(self, configs):
        self.train_flag = configs.train_flag
        self.path = os.path.join(os.path.join(os.path.join(configs.output_path, configs.dataset), configs.g_llm_model), self.train_flag)
            
        self.max_iterations = configs.max_iterations # maximum iterations for variant
        self.convergence_patience = getattr(configs, 'convergence_patience', 3)  # How many consecutive iterations with no change required for convergence, default 3
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

    def extract_numerical_value(self, text):
        """Extract numerical value from text"""
        if text is None:
            return None
        
        # First try direct conversion to float
        try:
            return float(text.strip())
        except ValueError:
            pass
        
        # Use regular expression to extract numerical value
        pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(pattern, text)
        
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                return None
        
        return None

    def check_convergence(self, prev_value, curr_value, prev_code, curr_code, tolerance=1e-6):
        """Check if converged (no significant change in value and code)"""
        # Check numerical convergence
        if prev_value is not None and curr_value is not None:
            # Ensure both values are numeric types
            try:
                prev_num = float(prev_value) if isinstance(prev_value, str) else prev_value
                curr_num = float(curr_value) if isinstance(curr_value, str) else curr_value
                if abs(prev_num - curr_num) < tolerance:
                    return True
            except (ValueError, TypeError):
                pass
        
        # Check code convergence (simple string comparison)
        if prev_code and curr_code:
            if prev_code.strip() == curr_code.strip():
                return True
        
        return False

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
                return content  # Return original content without attempting conversion
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
            float or None: Extracted numerical value as float, None if not found
        """
        if text is None:
            return None
        
        # First try direct conversion to float
        try:
            return float(text.strip())
        except ValueError:
            pass
        
        # Use regular expressions to extract numerical values
        import re
        # Find floating point numbers or integers in text
        pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(pattern, text)
        
        if matches:
            try:
                # Return the first numerical value found as float
                return float(matches[0])
            except ValueError:
                return None
        
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
                ground_truth_float = float(ground_truth)  # Directly try to convert ground_truth
                
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

    async def refine_variant(self, base_variant_path, problem_description, max_iterations, variant_name):
        """Improve a variant through multiple iterations with convergence-based stopping"""
        
        # Get initial variant data
        variant_data = self.get_variant_data(base_variant_path)
        
        # Create an improvement directory for this variant
        refinement_dir = os.path.join(os.path.dirname(base_variant_path), f"{variant_name}_refinements")
        if not os.path.exists(refinement_dir):
            os.makedirs(refinement_dir)
        
        # Track latest iteration data for continued improvement
        current_variant_data = variant_data
        
        # Maintain history for consecutive convergence checking
        history = []  # Store historical (code, value) pairs
        consecutive_same_count = 0  # Number of consecutive identical rounds
        
        # Get initial values for convergence checking
        initial_code = variant_data.get("code", "")
        initial_value = None
        if "optimal_value" in variant_data and variant_data["optimal_value"]:
            initial_value = self.extract_numerical_value(variant_data["optimal_value"])
        
        # Add initial state to history
        history.append((initial_code, initial_value))
        print(f"Initial state for {variant_name}: value = {initial_value}, patience = {self.convergence_patience}")
        
        # Start improvement iterations
        for iteration in range(1, max_iterations + 1):
            print(f"Starting {variant_name} refinement iteration {iteration}/{max_iterations}")
            
            # Create iteration directory
            iter_dir = os.path.join(refinement_dir, f"iteration_{iteration}")
            if not os.path.exists(iter_dir):
                os.makedirs(iter_dir)
            
            # Step 1: Multi-dimensional evaluation
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
            
            # Step 2: Generate improved code
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
            
            # Step 3: Execute code
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
                
                # Reset consecutive count on failure
                consecutive_same_count = 0
                history.append((code, None))  # Record failed state
                print(f"    🔄 {variant_name} iteration {iteration}: reset consecutive count due to failure")
                # Update variant data for next iteration
                current_variant_data = self.get_variant_data(iter_dir)
                continue
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
                    optimal_value = self.read_optimal_value(os.path.join(iter_dir, ref_optimal_path))
                    curr_value = self.extract_numerical_value(optimal_value)
                    
                    # Check consecutive convergence
                    current_state = (code, curr_value)
                    
                    # Check if same as previous state
                    if len(history) >= 1:
                        prev_code, prev_value = history[-1]
                        if self.check_convergence(prev_value, curr_value, prev_code, code):
                            consecutive_same_count += 1
                            print(f"    🔄 {variant_name} iteration {iteration}: consecutive same count = {consecutive_same_count}/{self.convergence_patience}")
                        else:
                            consecutive_same_count = 0
                            print(f"    📊 {variant_name} iteration {iteration}: value = {curr_value} (different from previous, reset count)")
                    
                    history.append(current_state)
                    
                    # Check if consecutive convergence requirement is met
                    if consecutive_same_count >= self.convergence_patience:
                        print(f"🔄 {variant_name} converged in iteration {iteration} (no significant change for {self.convergence_patience} consecutive iterations)")
                        return {
                            "status": "Converged",
                            "iterations": iteration,
                            "optimal_value": optimal_value,
                            "path": iter_dir,
                            "convergence_reason": f"no_change_for_{self.convergence_patience}_iterations",
                            "consecutive_same_count": consecutive_same_count,
                            "5element_description": current_variant_data.get("5element_description", ""),
                            "code": code,
                            "evaluation_result": current_variant_data.get("evaluation_result", "")
                        }
                    
                    print(f"    📊 {variant_name} iteration {iteration}: value = {curr_value}")
                else:
                    print(f"⚠️ {variant_name} iteration {iteration} did not produce ref_optimal_value.txt")
                    consecutive_same_count = 0  # Reset count
                    history.append((code, None))  # Record state without numerical value
                    print(f"    🔄 {variant_name} iteration {iteration}: reset consecutive count due to no optimal value")
            
            # Update variant data with current iteration results for next iteration
            current_variant_data = self.get_variant_data(iter_dir)
        
        # If we reach here, we've exhausted all iterations without convergence
        print(f"🔄 {variant_name} reached maximum iterations without convergence")
        return {
            "status": "Max_iterations",
            "iterations": max_iterations,
            "path": refinement_dir,
            "convergence_reason": "max_iterations",
            "consecutive_same_count": consecutive_same_count,
            "5element_description": current_variant_data.get("5element_description", ""),
            "code": current_variant_data.get("code", ""),
            "evaluation_result": current_variant_data.get("evaluation_result", ""),
            "optimal_value": current_variant_data.get("optimal_value", "")
        }

    def check_variant_correct(self, variant_path, ground_truth):
        """Check if a variant produces correct results"""
        optimal_path = os.path.join(variant_path, "ref_optimal_value.txt")
        if os.path.exists(optimal_path):
            optimal_value = self.read_optimal_value(optimal_path)
            return self.compare_with_ground_truth(optimal_value, ground_truth)
        return False

    def get_objective_direction(self, problem_dir):
        """Read objective direction from file"""
        objective_file = os.path.join(problem_dir, "objective_direction.txt")
        if os.path.exists(objective_file):
            with open(objective_file, 'r') as f:
                return f.read().strip()
        return None

    async def forward(self):
        """Improve solutions for all problems in the base directory with new aggregation strategy"""
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
                
                # Get problem description and objective direction
                problem_description = self.get_problem_description(problem_dir)
                objective_direction = self.get_objective_direction(problem_dir)
                
                if not problem_description:
                    print(f"Missing problem description for problem {problem_id}")
                    pbar.update(1)
                    continue
                
                print(f"Objective direction: {objective_direction}")
                
                # Check variants directory
                variants_dir = os.path.join(problem_dir, "variants")
                if not os.path.exists(variants_dir):
                    print(f"No variants directory found for problem {problem_id}")
                    pbar.update(1)
                    continue
                
                variant_dirs = [os.path.join(variants_dir, d) for d in sorted(os.listdir(variants_dir))
                    if (os.path.isdir(os.path.join(variants_dir, d)) and "refinements" not in d)]
                
                # Process each variant independently (no early stopping)
                print(f"Refining all {len(variant_dirs)} variants for problem {problem_id}...")
                
                # Track results for this problem
                problem_results = []
                
                # Process each variant independently
                for variant_dir in variant_dirs:
                    variant_name = os.path.basename(variant_dir)
                    print(f"\nRefining {variant_name} for problem {problem_id}")
                    
                    # Improve this variant with convergence-based stopping
                    refinement_result = await self.refine_variant(
                        variant_dir, 
                        problem_description, 
                        self.max_iterations,
                        variant_name
                    )
                    
                    # Add problem and variant information
                    refinement_result["problem_id"] = problem_id
                    refinement_result["variant"] = variant_name
                    
                    # Add to results
                    problem_results.append(refinement_result)
                    results.append(refinement_result)
                
                self.cache.append(problem_id)
                cache_file.write(problem_id + "\n") # append the id in cache_file

                # Create a summary file for improvements to this problem
                summary_path = os.path.join(refinement_dir, "refinement_summary.csv")
                problem_df = pd.DataFrame(problem_results)
                problem_df.to_csv(summary_path, index=False)
                
                # Update progress bar
                pbar.update(1)
        
        cache_file.close()
        
        # Compile and save summary results
        print(f"Debug: Total results collected: {len(results)}")
        if results:
            results_df = pd.DataFrame(results)
            print(f"Debug: DataFrame created with {len(results_df)} rows and columns: {list(results_df.columns)}")
        else:
            results_df = pd.DataFrame()
            print("Debug: No results collected, creating empty DataFrame")
        
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
        if len(results_df) > 0 and 'problem_id' in results_df.columns:
            print(f"Total problems processed: {len(set(results_df['problem_id']))}")
            print(f"Variants converged: {len(results_df[results_df['status'] == 'Converged'])}")
            print(f"Variants reached max iterations: {len(results_df[results_df['status'] == 'Max_iterations'])}")
        else:
            print("No results to summarize")
        
        return results_df
