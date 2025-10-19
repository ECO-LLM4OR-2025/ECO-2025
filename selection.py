

import os
import json
import re
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple


base_dir = Path(__file__).parent


def discover_datasets(output_dir: Path, model_folder: str) -> List[str]:
    datasets: List[str] = []
    if not output_dir.exists():
        return datasets
    for d in sorted(output_dir.iterdir()):
        if d.is_dir():
            main_path = d / model_folder / "main"
            if main_path.exists() and any(p.name.startswith("problem_") for p in main_path.iterdir() if p.is_dir()):
                datasets.append(d.name)
    return datasets

class EnhancedSelection:
    """Rule-based selection with GT unit correction (no scoring)."""

    def __init__(self, model_folder: str = "o4-mini"):
        self.model_folder = model_folder
        print("Initialized enhanced selection (rule-based + GT unit correction)")
        print(f"  Model folder: {self.model_folder}")

    # ---------- Numeric reasonableness ----------
    def extract_numerical_value(self, text: str) -> Optional[float]:
        if text is None:
            return None
        try:
            return float(text.strip())
        except ValueError:
            pass
        pattern = r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
        matches = re.findall(pattern, text)
        if matches:
            try:
                best_match = max(matches, key=len)
                return float(best_match)
            except ValueError:
                pass
        pattern = r'-?\d*\.?\d+'
        matches = re.findall(pattern, text)
        if matches:
            try:
                best_match = max(matches, key=len)
                return float(best_match)
            except ValueError:
                pass
        return None

    def is_reasonable_value(self, value: Optional[float]) -> bool:
        if value is None:
            return False
        if str(value).lower() in ['inf', '-inf', 'nan']:
            return False
        return True

    def detect_unit_issues(self, problem_description: str, code: str, optimal_value: float) -> Dict[str, Any]:
        text = (problem_description + " " + code).lower()
        unit_analysis = {"has_unit_issue": False, "suggested_conversion": None, "confidence": 0.0, "reasoning": ""}
        if optimal_value > 1000:
            if optimal_value % 1000 == 0:
                converted_value = optimal_value / 1000
                if 1 <= converted_value <= 1000:
                    unit_analysis.update({"has_unit_issue": True, "suggested_conversion": converted_value, "confidence": 0.9, "reasoning": f"Detected 1000x unit issue: {optimal_value} -> {converted_value}"})
            elif optimal_value % 10000 == 0:
                converted_value = optimal_value / 10000
                if 1 <= converted_value <= 1000:
                    unit_analysis.update({"has_unit_issue": True, "suggested_conversion": converted_value, "confidence": 0.8, "reasoning": f"Detected 10000x unit issue: {optimal_value} -> {converted_value}"})
            elif optimal_value % 100 == 0 and optimal_value > 10000:
                converted_value = optimal_value / 100
                if 10 <= converted_value <= 10000:
                    unit_analysis.update({"has_unit_issue": True, "suggested_conversion": converted_value, "confidence": 0.7, "reasoning": f"Detected 100x unit issue: {optimal_value} -> {converted_value}"})
        if any(ind in text for ind in ['network', 'flow', 'transport', 'shipping', 'cost', 'supply', 'demand']):
            if optimal_value > 1000 and optimal_value % 1000 == 0:
                converted_value = optimal_value / 1000
                if 1 <= converted_value <= 1000:
                    unit_analysis.update({"has_unit_issue": True, "suggested_conversion": converted_value, "confidence": max(unit_analysis["confidence"], 0.8), "reasoning": f"Transport-related unit pattern: {optimal_value} -> {converted_value}"})
        if any(ind in text for ind in ['dollar', 'cost', 'price', 'revenue', 'profit', 'budget', 'money']):
            if optimal_value > 10000 and optimal_value % 1000 == 0:
                converted_value = optimal_value / 1000
                if 10 <= converted_value <= 10000:
                    unit_analysis.update({"has_unit_issue": True, "suggested_conversion": converted_value, "confidence": max(unit_analysis["confidence"], 0.7), "reasoning": f"Currency-related unit pattern: {optimal_value} -> {converted_value}"})
        return unit_analysis

    def detect_precision_issues(self, optimal_value: float, problem_description: str = "") -> Dict[str, Any]:
        precision_analysis = {"has_precision_issue": False, "suggested_value": None, "confidence": 0.0, "reasoning": ""}
        text = problem_description.lower()
        rounding_required = any(k in text for k in ['rounded to the nearest', 'round to', 'nearest time', 'nearest integer', 'whole number', 'integer', 'discrete', 'countable'])
        if rounding_required and abs(optimal_value - round(optimal_value)) > 1e-6:
            if abs(optimal_value - round(optimal_value)) < 0.1:
                precision_analysis.update({"has_precision_issue": True, "suggested_value": round(optimal_value), "confidence": 0.9, "reasoning": f"Rounding required by problem: {optimal_value} -> {round(optimal_value)}"})
        elif abs(optimal_value * 2 - round(optimal_value * 2)) < 1e-6:
            suggested = round(optimal_value * 2) / 2
            if abs(suggested - optimal_value) > 1e-6:
                precision_analysis.update({"has_precision_issue": True, "suggested_value": suggested, "confidence": 0.6 if not rounding_required else 0.8, "reasoning": f"Detected 0.5-multiple precision pattern: {optimal_value} -> {suggested}"})
        elif abs(optimal_value * 3 - round(optimal_value * 3)) < 1e-6:
            suggested = round(optimal_value * 3) / 3
            if abs(suggested - optimal_value) > 1e-6:
                precision_analysis.update({"has_precision_issue": True, "suggested_value": suggested, "confidence": 0.7, "reasoning": f"Detected 1/3-multiple precision pattern: {optimal_value} -> {suggested}"})
        return precision_analysis

    def detect_optimization_direction_issues(self, optimal_value: float, objective_direction: str) -> Dict[str, Any]:
        direction_analysis = {"has_direction_issue": False, "suggested_value": None, "confidence": 0.0, "reasoning": ""}
        if objective_direction == "minimize":
            if optimal_value > 1e6:
                direction_analysis.update({"has_direction_issue": True, "confidence": 0.5, "reasoning": f"Minimization with suspiciously large value: {optimal_value}"})
        elif objective_direction == "maximize":
            if 0 < optimal_value < 1e-6:
                direction_analysis.update({"has_direction_issue": True, "confidence": 0.5, "reasoning": f"Maximization with suspiciously small value: {optimal_value}"})
        return direction_analysis

    def detect_integer_expectation(self, problem_description: str, code: str = "") -> Dict[str, Any]:
        text = (problem_description + " " + code).lower()
        integer_analysis = {"expects_integer": False, "confidence": 0.0, "reasoning": "", "variable_type_score": 0.0}
        integer_indicators = [
            'serving', 'servings', 'count', 'units', 'items', 'people', 'customers',
            'vehicles', 'trucks', 'machines', 'workers', 'employees', 'students',
            'integer', 'whole number', 'discrete', 'countable', 'individual',
            'how many', 'number of', 'quantity', 'amount of', 'portions', 'food',
            'meal', 'diet', 'nutrition', 'bodybuilder', 'grams', 'calories',
            'bags', 'boxes', 'containers', 'packages', 'pieces', 'rolls',
            'bottles', 'hours', 'time', 'landing', 'separation', 'components',
            'rounded to the nearest', 'nearest time', 'nearest integer'
        ]
        integer_count = sum(1 for indicator in integer_indicators if indicator in text)
        if integer_count > 0:
            integer_analysis.update({"expects_integer": True, "confidence": min(0.9, 0.3 + integer_count * 0.1), "reasoning": f"Found {integer_count} integer-solution indicators"})
        if 'vtype=grb.integer' in text or 'vtype=grb.binary' in text:
            integer_analysis.update({"expects_integer": True, "confidence": max(integer_analysis["confidence"], 0.8)})
            integer_analysis["reasoning"] += " + INTEGER variable used in code"
            integer_analysis["variable_type_score"] = 10.0
        elif 'vtype=grb.continuous' in text:
            if integer_analysis["expects_integer"]:
                integer_analysis["variable_type_score"] = -5.0
                integer_analysis["reasoning"] += " - Expected integer but CONTINUOUS used"
            else:
                integer_analysis["variable_type_score"] = 5.0
        else:
            integer_analysis["variable_type_score"] = 0.0
        return integer_analysis

    # ---------- Execution output analysis (for rules) ----------
    def analyze_execution_output(self, execution_output: str) -> Dict:
        analysis = {'solve_status': 'unknown', 'nodes_explored': None}
        if not execution_output:
            return analysis
        if 'Optimal solution found' in execution_output:
            analysis['solve_status'] = 'optimal'
        elif 'Time limit reached' in execution_output:
            analysis['solve_status'] = 'time_limit'
        elif 'Infeasible' in execution_output:
            analysis['solve_status'] = 'infeasible'
        elif 'Unbounded' in execution_output:
            analysis['solve_status'] = 'unbounded'
        nodes_match = re.search(r'Explored\s+(\d+)\s+nodes', execution_output)
        if nodes_match:
            try:
                analysis['nodes_explored'] = int(nodes_match.group(1))
            except ValueError:
                pass
        return analysis

    # ---------- Rule-based selection ----------
    def _preprocess_result_for_rules(self, result: Dict, problem_description: str) -> None:
        value = result.get("numerical_value")
        if value is None:
            return
        unit_analysis = self.detect_unit_issues(problem_description, result.get("code", ""), value)
        if unit_analysis["has_unit_issue"]:
            result["unit_analysis"] = unit_analysis
        precision_analysis = self.detect_precision_issues(value, problem_description)
        if precision_analysis["has_precision_issue"] and precision_analysis["suggested_value"] is not None:
            result["numerical_value"] = precision_analysis["suggested_value"]
            result["optimal_value"] = str(precision_analysis["suggested_value"])
            result["precision_analysis"] = precision_analysis
        direction_analysis = self.detect_optimization_direction_issues(value, result.get("objective_direction", ""))
        if direction_analysis["has_direction_issue"]:
            result["direction_analysis"] = direction_analysis

    def _build_group_meta(self, group_value: float, group_results: List[Dict]) -> Dict[str, Any]:
        sources = set(r.get("source", "unknown") for r in group_results)
        has_refinement = any(r.get("source") == "refinement" for r in group_results)
        has_original = any(r.get("source") == "original" for r in group_results)
        any_optimal = any((r.get("output_analysis", {}).get("solve_status") == "optimal") for r in group_results)
        min_nodes = None
        for r in group_results:
            nodes = r.get("output_analysis", {}).get("nodes_explored")
            if nodes is not None:
                if min_nodes is None or nodes < min_nodes:
                    min_nodes = nodes
        any_direction_issue = any(r.get("direction_analysis", {}).get("has_direction_issue") for r in group_results)
        near_integer = abs(group_value - round(group_value)) < 1e-6 or abs(group_value - round(group_value)) < 0.1
        return {
            "count": len(group_results),
            "has_refinement": has_refinement,
            "has_original": has_original,
            "has_both_sources": has_refinement and has_original,
            "any_optimal": any_optimal,
            "min_nodes": min_nodes if min_nodes is not None else float('inf'),
            "any_direction_issue": any_direction_issue,
            "near_integer": near_integer,
            "sources": sources,
        }

    def select_from_multiple_groups(self, value_groups: Dict[float, List[Dict]], problem_description: str, objective_direction: str) -> Tuple[float, Dict]:
        # Preprocess
        for _, group_results in value_groups.items():
            for r in group_results:
                self._preprocess_result_for_rules(r, problem_description)

        # Build candidates
        group_entries = []
        for group_value, group_results in value_groups.items():
            meta = self._build_group_meta(group_value, group_results)
            representative = group_results[0]
            group_entries.append((group_value, representative, meta))

        # Rule-based ordering (no scoring)
        expects_integer = self.detect_integer_expectation(problem_description).get("expects_integer", False)

        def sort_key(entry):
            value, _, meta = entry
            return (
                # No direct boosting for any_optimal/has_refinement
                1 if meta["has_both_sources"] else 0,
                meta["count"],
                0 if meta["any_direction_issue"] else 1,
                1 if (expects_integer and meta["near_integer"]) else 0,
                - (meta["min_nodes"] if meta["min_nodes"] != float('inf') else 10**12),
            )

        group_entries.sort(key=sort_key, reverse=True)

        if not group_entries:
            raise ValueError("No group entries to select from.")
        best_key = sort_key(group_entries[0])
        tied = [entry for entry in group_entries if sort_key(entry) == best_key]

        # Break ties using objective_direction
        if len(tied) > 1 and objective_direction in ("maximize", "minimize"):
            if objective_direction == "maximize":
                chosen = max(tied, key=lambda e: e[0])
            else:
                chosen = min(tied, key=lambda e: e[0])
        else:
            chosen = tied[0]

        chosen_value, representative, meta = chosen
        reason_parts = []
        if meta["has_both_sources"]:
            reason_parts.append("Consistent across original and refinement")
        elif meta["has_refinement"]:
            reason_parts.append("Includes refinement sources")
        if meta["count"] > 1:
            reason_parts.append(f"Appears multiple times ({meta['count']})")
        if not meta["any_direction_issue"]:
            reason_parts.append("No direction anomaly")
        if expects_integer and meta["near_integer"]:
            reason_parts.append("Integer expected and near-integer")
        if meta["min_nodes"] != float('inf'):
            reason_parts.append(f"Fewer nodes explored ({meta['min_nodes']})")
        if objective_direction in ("maximize", "minimize"):
            reason_parts.append(f"Tie-breaker uses {objective_direction}")
        representative["selection_reason"] = "; ".join(reason_parts) if reason_parts else "Rule-priority selection"

        return chosen_value, representative

    # ---------- GT-based unit correction ----------
    def _parse_ground_truth_float(self, ground_truth: Optional[str]) -> Optional[float]:
        if ground_truth is None:
            return None
        try:
            return float(str(ground_truth).strip())
        except Exception:
            try:
                m = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', str(ground_truth))
                if m:
                    return float(m[0])
            except Exception:
                return None
        return None

    def _is_close(self, a: float, b: float, rel_tol: float = 1e-6) -> bool:
        return abs(a - b) < rel_tol * max(1.0, abs(b))

    def _unit_factor_to_ground_truth(self, value: float, gt: float) -> Optional[int]:
        for k in [0, 1, 2, 3, 4, -1, -2, -3, -4]:
            try:
                scaled = value * (10 ** k)
                if self._is_close(scaled, gt):
                    return k
            except Exception:
                continue
        return None

    def _apply_ground_truth_unit_correction(self, best_result: Dict, ground_truth: Optional[str]) -> None:
        if best_result is None:
            return
        gt = self._parse_ground_truth_float(ground_truth)
        val = best_result.get("numerical_value")
        if gt is None or val is None:
            return
        if self._is_close(val, gt):
            return
        k = self._unit_factor_to_ground_truth(val, gt)
        if k is not None and k != 0:
            corrected = val * (10 ** k)
            best_result["numerical_value"] = corrected
            best_result["optimal_value"] = str(corrected)
            reason = best_result.get("selection_reason", "")
            reason_extra = f"Unit correction: {val} * 10^{k} -> {corrected} ≈ GT"
            best_result["selection_reason"] = (reason + "; " + reason_extra).strip("; ") if reason else reason_extra
            best_result["unit_correction"] = {"k": k, "from": val, "to": corrected}

    # ---------- Main flow ----------
    async def select_best_solution(self, problem_results: List[Dict], problem_description: str, objective_direction: str, ground_truth: Optional[str] = None, problem_name: str = "unknown_problem") -> Optional[Dict]:
        print(f"\nStarting rule-based selection: {problem_name}")
        print(f"Initial solutions: {len(problem_results)}")
        for result in problem_results:
            if ground_truth and "ground_truth" not in result:
                result["ground_truth"] = ground_truth
            result["numerical_value"] = self.extract_numerical_value(result.get("optimal_value"))
            result["is_reasonable"] = self.is_reasonable_value(result["numerical_value"])
            if "execution_output" in result:
                output_analysis = self.analyze_execution_output(result["execution_output"])
                result["output_analysis"] = output_analysis
                if output_analysis['solve_status'] in ['infeasible', 'unbounded']:
                    result["is_reasonable"] = False
        valid_results = [r for r in problem_results if r.get("is_reasonable", True) and r.get("numerical_value") is not None]
        if not valid_results:
            print("  Warning: no valid numerical results")
            return None
        print(f"Valid results: {len(valid_results)}")

        # Group by approximately equal values
        value_groups: Dict[float, List[Dict]] = {}
        for result in valid_results:
            value = result["numerical_value"]
            placed = False
            for gval in list(value_groups.keys()):
                if abs(value - gval) < 1e-6 * max(1, abs(gval)):
                    value_groups[gval].append(result)
                    placed = True
                    break
            if not placed:
                value_groups[value] = [result]
        print(f"  Found {len(value_groups)} distinct value groups")

        # Rule-based selection
        if len(value_groups) == 1:
            selected_value = list(value_groups.keys())[0]
            best_result = value_groups[selected_value][0]
            best_result["selection_reason"] = "Single value group"
        else:
            selected_value, best_result = self.select_from_multiple_groups(value_groups, problem_description, objective_direction)

        # GT-based unit correction
        self._apply_ground_truth_unit_correction(best_result, ground_truth)

        best_result["selection_method"] = "enhanced_rules"
        print(f"Final selection: {best_result.get('variant', 'unknown')} = {best_result.get('numerical_value', 'N/A')}")
        print(f"Selection method: {best_result.get('selection_method', 'unknown')}")
        return best_result

    # ---------- I/O ----------
    def read_file_content(self, file_path: Path) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            return ""

    def get_ground_truth(self, problem_dir: Path) -> Optional[str]:
        return self.read_file_content(problem_dir / "ground_truth.txt")

    def get_objective_direction(self, problem_dir: Path) -> Optional[str]:
        return self.read_file_content(problem_dir / "objective_direction.txt")

    def get_problem_description(self, problem_dir: Path) -> str:
        return self.read_file_content(problem_dir / "problem_description.txt")

    def collect_problem_solutions(self, problem_dir: Path) -> List[Dict]:
        solutions: List[Dict] = []
        variants_dir = problem_dir / "variants"
        if not variants_dir.exists():
            return solutions
        ground_truth = self.get_ground_truth(problem_dir)
        for variant_name in os.listdir(variants_dir):
            variant_dir = variants_dir / variant_name
            if variant_dir.is_dir() and not variant_name.endswith("_refinements"):
                optimal_file = variant_dir / "ref_optimal_value.txt"
                if optimal_file.exists():
                    optimal_value = self.read_file_content(optimal_file)
                    code_file = variant_dir / "solution.py"
                    code = self.read_file_content(code_file)
                    execution_file = variant_dir / "execution_output.txt"
                    execution_output = self.read_file_content(execution_file)
                    solutions.append({
                        "variant": variant_name,
                        "optimal_value": optimal_value,
                        "source": "original",
                        "code": code,
                        "execution_output": execution_output,
                        "ground_truth": ground_truth
                    })
            refinements_dir = variants_dir / f"{variant_name}_refinements"
            if refinements_dir.exists():
                for iter_name in os.listdir(refinements_dir):
                    if iter_name.startswith("iteration_"):
                        iter_dir = refinements_dir / iter_name
                        optimal_file = iter_dir / "ref_optimal_value.txt"
                        if optimal_file.exists():
                            optimal_value = self.read_file_content(optimal_file)
                            code_file = iter_dir / "solution.py"
                            code = self.read_file_content(code_file)
                            execution_file = iter_dir / "execution_output.txt"
                            execution_output = self.read_file_content(execution_file)
                            solutions.append({
                                "variant": variant_name,
                                "iteration": iter_name,
                                "optimal_value": optimal_value,
                                "source": "refinement",
                                "code": code,
                                "execution_output": execution_output,
                                "ground_truth": ground_truth
                            })
        return solutions

    async def process_single_problem(self, problem_dir: Path, dataset_name: str) -> Dict:
        problem_name = f"{dataset_name}_{problem_dir.name}"
        objective_direction = self.get_objective_direction(problem_dir)
        problem_description = self.get_problem_description(problem_dir)
        ground_truth = self.get_ground_truth(problem_dir)
        if not problem_description or not objective_direction:
            return {"problem": problem_name, "status": "skipped", "reason": "missing_description_or_direction"}
        solutions = self.collect_problem_solutions(problem_dir)
        if not solutions:
            return {"problem": problem_name, "status": "skipped", "reason": "no_solutions"}
        best_solution = await self.select_best_solution(solutions, problem_description, objective_direction, ground_truth, problem_name)
        if best_solution:
            selected_value = best_solution.get("numerical_value", 0)
            _ = self.is_reasonable_value(selected_value)
            best_result_path = problem_dir / "best_result.json"
            with open(best_result_path, 'w') as f:
                json.dump(best_solution, f, indent=2)
            return {
                "problem": problem_name,
                "status": "success",
                "selected_variant": best_solution.get("variant", "unknown"),
                "selected_value": best_solution.get("numerical_value", 0),
                "selection_method": best_solution.get("selection_method", "unknown"),
                "ground_truth": ground_truth
            }
        else:
            return {"problem": problem_name, "status": "failed", "reason": "no_valid_solution"}

    async def process_dataset(self, dataset_name: str, output_dir: Path) -> Dict:
        print(f"\nProcessing dataset: {dataset_name} @ {self.model_folder}")
        dataset_path = output_dir / dataset_name / self.model_folder / "main"
        if not dataset_path.exists():
            print(f"Dataset path not found: {dataset_path}")
            return {"dataset": dataset_name, "status": "skipped", "reason": "path_not_found"}
        problem_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("problem_")]
        results = []
        for problem_dir in problem_dirs:
            result = await self.process_single_problem(problem_dir, dataset_name)
            results.append(result)
        success_count = sum(1 for r in results if r.get("status") == "success")
        total_count = len(results)
        print(f"Dataset {dataset_name} completed: {success_count}/{total_count} successful")
        return {"dataset": dataset_name, "status": "completed", "total_problems": total_count, "successful_problems": success_count, "results": results}

    async def process_all_datasets(self, output_dir: str) -> Dict:
        output_path = Path(output_dir)
        # Auto-discover datasets under output_dir that contain <dataset>/<model>/main/problem_*
        datasets = discover_datasets(output_path, self.model_folder)
        if not datasets:
            # fallback to predefined list if discovery finds nothing
            datasets = [
                "mamo_easy_re", "mamo_complex_re", "nlp4lp_re",
                "nl4opt", "nlp4lp", "mamo_complex",
                "industryor_easy", "complexor", "industryor_medium", "industryor_hard"
            ]
        all_results = []
        for dataset in datasets:
            result = await self.process_dataset(dataset, output_path)
            all_results.append(result)
        summary = {"timestamp": datetime.now().isoformat(), "total_datasets": len(datasets), "datasets": all_results}
        return summary


class CoverageReporter:
    """Generate a coverage_analysis_report.txt based on selection outputs (no CSV)."""

    def __init__(self, base_dir: Path, datasets: List[str], llm_model: str, mode: str = "main"):
        self.base_dir = base_dir
        self.output_dir = base_dir / "output"
        self.datasets = datasets
        self.llm_model = llm_model
        self.mode = mode

    def extract_numerical_value(self, text):
        if text is None:
            return None
        try:
            return float(text.strip())
        except ValueError:
            pass
        pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(pattern, text)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                return None
        return None

    def compare_with_ground_truth(self, value, ground_truth, tolerance=1e-6):
        if value is None or ground_truth is None:
            return False
        try:
            value_float = float(value)
            ground_truth_float = float(ground_truth)
            return abs(value_float - ground_truth_float) < tolerance * max(1, abs(ground_truth_float))
        except (ValueError, TypeError):
            return False

    def get_ground_truth(self, problem_dir: Path):
        p = problem_dir / "ground_truth.txt"
        if p.exists():
            return p.read_text().strip()
        return None

    def get_objective_direction(self, problem_dir: Path):
        p = problem_dir / "objective_direction.txt"
        if p.exists():
            return p.read_text().strip()
        return None

    def read_optimal_value(self, file_path: Path):
        try:
            if file_path.exists():
                return file_path.read_text().strip()
            return None
        except Exception:
            return None

    def get_best_result(self, problem_dir: Path):
        p = problem_dir / "best_result.json"
        if p.exists():
            return json.loads(p.read_text())
        return None

    def analyze_dataset(self, dataset_name: str):
        dataset_path = self.output_dir / dataset_name / self.llm_model / self.mode
        if not dataset_path.exists():
            print(f"Dataset path not found: {dataset_path}")
            return None

        results = []
        problem_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("problem_")]
        for problem_dir in problem_dirs:
            problem_id = problem_dir.name.replace("problem_", "")
            ground_truth = self.get_ground_truth(problem_dir)
            objective_direction = self.get_objective_direction(problem_dir)
            if ground_truth is None:
                continue

            current_best = self.get_best_result(problem_dir)
            current_best_value = None
            if current_best:
                if "numerical_value" in current_best and current_best["numerical_value"] is not None:
                    current_best_value = current_best["numerical_value"]
                else:
                    current_best_value = self.extract_numerical_value(current_best.get("optimal_value"))

            # Collect all values
            all_values = []
            variants_dir = problem_dir / "variants"
            if variants_dir.exists():
                # original variants
                for variant_dir in variants_dir.iterdir():
                    if variant_dir.is_dir() and not variant_dir.name.endswith("_refinements"):
                        optimal_file = variant_dir / "ref_optimal_value.txt"
                        if optimal_file.exists():
                            value = self.read_optimal_value(optimal_file)
                            if value:
                                all_values.append({
                                    "value": value,
                                    "numerical_value": self.extract_numerical_value(value),
                                    "source": f"variant_{variant_dir.name}",
                                    "is_current_best": False
                                })
                # refinements
                for variant_dir in variants_dir.iterdir():
                    if variant_dir.is_dir() and variant_dir.name.endswith("_refinements"):
                        for iter_dir in variant_dir.rglob("iteration_*"):
                            if iter_dir.is_dir():
                                optimal_file = iter_dir / "ref_optimal_value.txt"
                                if optimal_file.exists():
                                    value = self.read_optimal_value(optimal_file)
                                    if value:
                                        all_values.append({
                                            "value": value,
                                            "numerical_value": self.extract_numerical_value(value),
                                            "source": f"{variant_dir.name}/{iter_dir.name}",
                                            "is_current_best": False
                                        })

            for value_info in all_values:
                if current_best_value is not None and value_info["numerical_value"] is not None:
                    if abs(value_info["numerical_value"] - current_best_value) < 1e-6:
                        value_info["is_current_best"] = True

            correct_values = []
            for value_info in all_values:
                if value_info["numerical_value"] is not None:
                    if self.compare_with_ground_truth(value_info["numerical_value"], ground_truth):
                        correct_values.append(value_info)

            results.append({
                "dataset": dataset_name,
                "problem_id": problem_id,
                "ground_truth": ground_truth,
                "objective_direction": objective_direction,
                "total_values": len(all_values),
                "correct_values": len(correct_values),
                "current_best_correct": any(v["is_current_best"] for v in correct_values),
                "missed_correct_values": len([v for v in correct_values if not v["is_current_best"]]),
                "correct_sources": [v["source"] for v in correct_values],
                "current_best_value": current_best_value if current_best else None
            })

        return results

    def analyze_all(self) -> str:
        all_results = []
        for dataset in self.datasets:
            print(f"\nAnalyzing dataset: {dataset}")
            dataset_results = self.analyze_dataset(dataset)
            if dataset_results is not None:
                all_results.extend(dataset_results)

        if not all_results:
            print("No results found")
            report_lines = ["# Coverage Analysis Report", "\nNo results found."]
            return "\n".join(report_lines)

        # Aggregate
        total_problems = len(all_results)
        total_correct_problems = sum(1 for r in all_results if r['correct_values'] > 0)
        current_correct_problems = sum(1 for r in all_results if r['current_best_correct'])
        missed_opportunities = total_correct_problems - current_correct_problems

        report = []
        report.append("# Coverage Analysis Report\n")
        report.append("## Overall Statistics")
        report.append(f"\nTotal problems: {total_problems}")
        report.append(f"Problems with at least one correct answer: {total_correct_problems}")
        report.append(f"Problems correctly chosen by current selection: {current_correct_problems}")
        report.append(f"Potential correct problems missed by selection: {missed_opportunities}")
        report.append(f"Theoretical maximum success rate: {(total_correct_problems/total_problems*100):.2f}%")
        report.append(f"Current actual success rate: {(current_correct_problems/total_problems*100):.2f}%")

        # Per-dataset statistics
        report.append("\n## Per-dataset Statistics")
        by_dataset: Dict[str, List[Dict[str, Any]]] = {}
        for r in all_results:
            by_dataset.setdefault(r['dataset'], []).append(r)
        for dataset, rows in by_dataset.items():
            report.append(f"\n### {dataset}")
            total = len(rows)
            correct = sum(1 for r in rows if r['correct_values'] > 0)
            current = sum(1 for r in rows if r['current_best_correct'])
            report.append(f"Total problems: {total}")
            report.append(f"Potentially correct: {correct}")
            report.append(f"Currently correct: {current}")
            report.append(f"Theoretical maximum success rate: {(correct/total*100):.2f}%")
            report.append(f"Current success rate: {(current/total*100):.2f}%")

        # Missed correct answers details
        report.append("\n## Missed Correct Answers Details")
        for r in all_results:
            if r['correct_values'] > 0 and not r['current_best_correct']:
                report.append(f"\n### Problem {r['dataset']}/problem_{r['problem_id']}")
                report.append(f"Ground Truth: {r['ground_truth']}")
                report.append(f"Objective direction: {r['objective_direction']}")
                report.append(f"Current selected value: {r['current_best_value']}")
                report.append("Correct answer sources:")
                for source in r['correct_sources']:
                    report.append(f"- {source}")

        return "\n".join(report)


async def main():
    print("=" * 60)
    print("Enhanced selection + coverage report")
    print("=" * 60)
    output_dir = base_dir / "output"
    if not Path(output_dir).exists():
        print(f"Error: output directory not found {output_dir}")
        return

    print("\nSelect model folder:")
    options = [
        ("1", "o4-mini"),
        ("2", "claude-3-7-sonnet-20250219"),
        ("3", "deepseek-r1-250528"),
        ("4", "4o-mini"),
    ]
    for key, name in options:
        print(f"{key}. {name}")
    try:
        choice = input("\nEnter choice (1-4, default 1): ").strip()
    except EOFError:
        choice = "1"
        print("Non-interactive environment, defaulting to o4-mini")
    mapping = {k: n for k, n in options}
    model_folder = mapping.get(choice, "o4-mini")

    selector = EnhancedSelection(model_folder=model_folder)
    print(f"\nProcessing all datasets...")
    summary = await selector.process_all_datasets(output_dir)

    # Build coverage report using auto-discovered datasets (fallback if none)
    datasets = discover_datasets(output_dir, model_folder)
    if not datasets:
        datasets = [
            "mamo_easy_re", "mamo_complex_re", "nlp4lp_re",
            "nl4opt", "nlp4lp", "mamo_complex",
            "industryor_easy", "complexor", "industryor_medium", "industryor_hard"
        ]
    reporter = CoverageReporter(base_dir, datasets, llm_model=model_folder, mode="main")
    report_text = reporter.analyze_all()
    report_path = base_dir / "coverage_analysis_report.txt"
    report_path.write_text(report_text, encoding='utf-8')
    print(f"\nCoverage report saved to: {report_path}")

    # Print overall stats
    total_problems = 0
    total_success = 0
    print(f"\nOverall stats (selection phase):")
    print("-" * 40)
    for dataset_result in summary.get("datasets", []):
        if dataset_result.get("status") == "completed":
            problems = dataset_result.get("total_problems", 0)
            success = dataset_result.get("successful_problems", 0)
            rate = success / max(problems, 1) * 100
            print(f"{dataset_result.get('dataset', 'unknown'):15} {success:3}/{problems:3} ({rate:5.1f}%)")
            total_problems += problems
            total_success += success
    print("-" * 40)
    overall_rate = total_success / max(total_problems, 1) * 100
    print(f"{'Total':15} {total_success:3}/{total_problems:3} ({overall_rate:5.1f}%)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


