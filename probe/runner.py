import os
import ast
import json
import asyncio
import tempfile
import subprocess
from typing import Any
from dataclasses import asdict

# TODO: probably simplify this?

from .generator import Generator
from .grounding import ConstraintExtractor
from .llm import LLMClient
from .models import CounterExecutionResult, CounterImplementation, ProbeRunResult, PropertyAttempt, PropertyCandidate, TestingTask
from .planner import Planner
from .repository import RepositoryIndex
from .semantic_graph import SemanticGraphBuilder
from .validator import Validator


class ProbeRunner:
    def __init__(self, llm: LLMClient, python_executable: str, repo_root: str | None = None, max_fix_retries: int = 3, max_refinement_rounds: int = 3, max_initial_properties: int = 3, enforce_strength_checks: bool = True, reject_weak_suites: bool = True, pytest_timeout_seconds: int = 120):
        self.llm = llm
        self.python_executable = python_executable
        self.repo_root = repo_root
        self.max_fix_retries = max_fix_retries
        self.max_property_replans = max(3, self.max_fix_retries + 1)
        self.max_refinement_rounds = max_refinement_rounds
        self.max_initial_properties = max_initial_properties
        self.reject_weak_suites = reject_weak_suites

        self.constraint_extractor = ConstraintExtractor()
        self.planner = Planner(llm)
        self.validator = Validator(llm)
        self.generator = Generator(llm, python_executable, repo_root, max_fix_retries, enforce_strength_checks=enforce_strength_checks, pytest_timeout_seconds=pytest_timeout_seconds)

    def run(self, api_info: dict[str, Any], repository_index: RepositoryIndex):
        target = repository_index.find_function(api_info)
        cons = self.constraint_extractor.extract(target)
        graph = SemanticGraphBuilder(repository_index)
        structs, edge_map = graph.collect_candidates(target, max_candidates=12)
        cand_edges = [edge for edges in edge_map.values() for edge in edges]
        sel_names = self.planner.select_context_functions(target=target, constraints=cons, candidate_functions=structs, semantic_edges=cand_edges, max_functions=4)
        sel_set = set(sel_names)
        ctx_funcs = [item for item in structs if item.full_name in sel_set][:4]
        sem_edges = [edge for item in ctx_funcs for edge in edge_map.get(item.full_name, [])]

        #  面向过程编程...
        valid_props: list[str] = []
        pbt_codes: list[str] = []
        prop_tries: list[PropertyAttempt] = []
        tried_props: set[str] = set()
        survivors: list[CounterImplementation] = []
        checks: list[dict[str, Any]] = []
        rounds = 0
        tgt_budget = max(1, self.max_initial_properties)
        x_budget = 1 if ctx_funcs else 0
        prop_budget = tgt_budget + x_budget
        max_tries = max(prop_budget * 2, prop_budget + self.max_refinement_rounds + 1)
        phases = [
            ("target_only", [], [], ctx_funcs, sem_edges, tgt_budget, True),
        ]
        if x_budget:
            phases.append(("cross_function", ctx_funcs, sem_edges, ctx_funcs, sem_edges, x_budget, False))

        for (
            mode,
            phase_ctx,
            phase_edges,
            task_ctx,
            task_edges,
            budget,
            allow_adv,
        ) in phases:
            if budget <= 0: continue
            pending: list[PropertyCandidate] = []
            adv_props: set[str] = set()
            start_valid = len(valid_props)
            plan_q = 0
            max_plan_q = 2 if mode == "target_only" else 1

            while len(prop_tries) < max_tries:
                phase_valid = len(valid_props) - start_valid
                next_adv = bool(pending and pending[0].property_text in adv_props)
                if phase_valid >= budget and not next_adv:
                    break
                if not pending:
                    remaining = budget - phase_valid
                    if remaining <= 0 or plan_q >= max_plan_q:
                        break
                    cands = self.planner.plan_properties(target=target, constraints=cons, context_functions=phase_ctx, semantic_edges=phase_edges, existing_properties=sorted(set(valid_props) | tried_props), max_properties=max(remaining, 1), planning_mode=mode)
                    plan_q += 1
                    self._enqueue_candidates(pending, cands, valid_props, tried_props)
                    if not pending:
                        break

                cand = pending.pop(0)
                adv_props.discard(cand.property_text)
                tried_props.add(cand.property_text)

                task = self._build_task(target, cons, task_ctx, task_edges, cand)
                prop_result = self._process_task(task, valid_props, pbt_codes)
                prop_tries.append(prop_result)


                # print(property_result.status)
                if prop_result.status == "library_bug":
                    return ProbeRunResult(status='library_bug_found', target_function=target.full_name, final_pbt_code=self.generator.combine_pbt_codes(pbt_codes, task), properties=prop_tries, bug_report={'property': prop_result.property_text, 'reasoning': prop_result.reasoning, 'error_message': prop_result.error_message, 'code': prop_result.code}, strengthening_rounds=rounds, adversarial_checks=checks)

                # print(property_result)

                if prop_result.status != "passed" or not prop_result.code:
                    continue

                valid_props.append(prop_result.property_text)
                pbt_codes.append(prop_result.code)

                if not allow_adv:
                    continue

                combo_code = self.generator.combine_pbt_codes(pbt_codes, task)
                survivors, survivor = self._recheck_surviving_counterexamples(task, combo_code, survivors, checks)
                if survivor is None:
                    ctr_impls = self.validator.generate_counter_implementations(task, combo_code, max_candidates=3)
                    for ctr_impl in ctr_impls:
                        ctr_exec = self._execute_counter_implementation(task, combo_code, ctr_impl)
                        checks.append(self._adversarial_check_record(task=task, counter_implementation=ctr_impl, counter_execution=ctr_exec, source='generated'))
                        if ctr_exec.survived:
                            survivors = self._remember_surviving_counterexample(survivors, ctr_impl)
                            if survivor is None:
                                survivor = (ctr_impl, ctr_exec)
                if survivor is None or rounds >= self.max_refinement_rounds:
                    continue
                ctr_impl, ctr_exec = survivor
                next_prop = self.planner.strengthen_property(target=target, constraints=cons, context_functions=task_ctx, semantic_edges=task_edges, existing_properties=valid_props, counter_implementation=ctr_impl, counter_execution=ctr_exec)
                rounds += 1
                added = self._enqueue_candidates(pending, [next_prop] if next_prop is not None else [], valid_props, tried_props, front=True)
                adv_props.update(cand.property_text for cand in added)

        final = self.generator.combine_pbt_codes(pbt_codes, target) if pbt_codes else None
        if final and self.reject_weak_suites:
            suite = self._final_suite_issues(target, prop_tries, final)
            if suite:
                return ProbeRunResult(status='no_properties', target_function=target.full_name, final_pbt_code=None, properties=prop_tries, strengthening_rounds=rounds, bug_report={'suite_rejected': True, 'issues': suite}, adversarial_checks=checks)
        status = "success" if final else "no_properties"
        return ProbeRunResult(status=status, target_function=target.full_name, final_pbt_code=final, properties=prop_tries, strengthening_rounds=rounds, adversarial_checks=checks)

    async def arun(self, api_info: dict[str, Any], repository_index: RepositoryIndex):
        target = repository_index.find_function(api_info)
        cons = self.constraint_extractor.extract(target)
        graph = SemanticGraphBuilder(repository_index)
        structs, edge_map = graph.collect_candidates(target, max_candidates=12)
        cand_edges = [edge for edges in edge_map.values() for edge in edges]
        sel_names = await self.planner.aselect_context_functions(target=target, constraints=cons, candidate_functions=structs, semantic_edges=cand_edges, max_functions=4)
        sel_set = set(sel_names)
        ctx_funcs = [item for item in structs if item.full_name in sel_set][:4]
        sem_edges = [edge for item in ctx_funcs for edge in edge_map.get(item.full_name, [])]

        valid_props: list[str] = []
        pbt_codes: list[str] = []
        prop_tries: list[PropertyAttempt] = []
        tried_props: set[str] = set()
        survivors: list[CounterImplementation] = []
        checks: list[dict[str, Any]] = []
        rounds = 0
        tgt_budget = max(1, self.max_initial_properties)
        x_budget = 1 if ctx_funcs else 0
        prop_budget = tgt_budget + x_budget
        max_tries = max(prop_budget * 2, prop_budget + self.max_refinement_rounds + 1)
        phases = [
            ("target_only", [], [], ctx_funcs, sem_edges, tgt_budget, True),
        ]
        if x_budget:
            phases.append(("cross_function", ctx_funcs, sem_edges, ctx_funcs, sem_edges, x_budget, False))

        for (
            mode,
            phase_ctx,
            phase_edges,
            task_ctx,
            task_edges,
            budget,
            allow_adv,
        ) in phases:
            if budget <= 0: continue

            pending: list[PropertyCandidate] = []
            adv_props: set[str] = set()
            start_valid = len(valid_props)
            plan_q = 0
            max_plan_q = 2 if mode == "target_only" else 1

            while len(prop_tries) < max_tries:
                phase_valid = len(valid_props) - start_valid
                next_adv = bool(pending and pending[0].property_text in adv_props)
                if phase_valid >= budget and not next_adv:
                    break
                if not pending:
                    remaining = budget - phase_valid
                    if remaining <= 0 or plan_q >= max_plan_q:
                        break
                    cands = await self.planner.aplan_properties(target=target, constraints=cons, context_functions=phase_ctx, semantic_edges=phase_edges, existing_properties=sorted(set(valid_props) | tried_props), max_properties=max(remaining, 1), planning_mode=mode)
                    plan_q += 1
                    self._enqueue_candidates(pending, cands, valid_props, tried_props)
                    if not pending:
                        break

                cand = pending.pop(0)
                adv_props.discard(cand.property_text)
                tried_props.add(cand.property_text)

                task = self._build_task(target, cons, task_ctx, task_edges, cand)
                prop_result = await self._aprocess_task(task, valid_props, pbt_codes)
                prop_tries.append(prop_result)

                if prop_result.status == "library_bug":
                    return ProbeRunResult(status='library_bug_found', target_function=target.full_name, final_pbt_code=self.generator.combine_pbt_codes(pbt_codes, task), properties=prop_tries, bug_report={'property': prop_result.property_text, 'reasoning': prop_result.reasoning, 'error_message': prop_result.error_message, 'code': prop_result.code}, strengthening_rounds=rounds, adversarial_checks=checks)

                if prop_result.status != "passed" or not prop_result.code:
                    continue

                valid_props.append(prop_result.property_text)
                pbt_codes.append(prop_result.code)

                if not allow_adv:
                    # 跳过
                    continue

                combo_code = self.generator.combine_pbt_codes(pbt_codes, task)
                survivors, survivor = await asyncio.to_thread(self._recheck_surviving_counterexamples, task, combo_code, survivors, checks)
                if survivor is None:
                    ctr_impls = await self.validator.agenerate_counter_implementations(task, combo_code, max_candidates=3)
                    for ctr_impl in ctr_impls:
                        ctr_exec = await asyncio.to_thread(self._execute_counter_implementation, task, combo_code, ctr_impl)
                        checks.append(self._adversarial_check_record(task=task, counter_implementation=ctr_impl, counter_execution=ctr_exec, source='generated'))
                        if ctr_exec.survived:
                            survivors = self._remember_surviving_counterexample(survivors, ctr_impl)
                            if survivor is None:
                                survivor = (ctr_impl, ctr_exec)
                if survivor is None or rounds >= self.max_refinement_rounds:
                    continue
                ctr_impl, ctr_exec = survivor
                next_prop = await self.planner.astrengthen_property(target=target, constraints=cons, context_functions=task_ctx, semantic_edges=task_edges, existing_properties=valid_props, counter_implementation=ctr_impl, counter_execution=ctr_exec)
                rounds += 1
                added = self._enqueue_candidates(pending, [next_prop] if next_prop is not None else [], valid_props, tried_props, front=True)
                adv_props.update(cand.property_text for cand in added)

        final = self.generator.combine_pbt_codes(pbt_codes, target) if pbt_codes else None
        if final and self.reject_weak_suites:
            suite = self._final_suite_issues(target, prop_tries, final)
            if suite:
                return ProbeRunResult(status='no_properties', target_function=target.full_name, final_pbt_code=None, properties=prop_tries, strengthening_rounds=rounds, bug_report={'suite_rejected': True, 'issues': suite}, adversarial_checks=checks)
        status = "success" if final else "no_properties"
        return ProbeRunResult(status=status, target_function=target.full_name, final_pbt_code=final, properties=prop_tries, strengthening_rounds=rounds, adversarial_checks=checks)

    def _enqueue_candidates(self, pending_candidates: list[PropertyCandidate], candidates: list[PropertyCandidate], validated_properties: list[str], attempted_properties: set[str], front: bool = False):
        if not candidates:
            return []
        seen = {cand.property_text for cand in pending_candidates}
        seen.update(validated_properties)
        seen.update(attempted_properties)
        fresh = [cand for cand in candidates if cand.property_text not in seen]
        if front:
            pending_candidates[:0] = fresh
            return fresh
        pending_candidates.extend(fresh)
        return fresh

    @staticmethod
    def _canonical_property_text(text: str):
        return " ".join(text.split())

    def _extend_replan_history(self, replan_history: tuple[str, ...], property_text: str):
        canonical = self._canonical_property_text(property_text)
        for existing in replan_history:
            if self._canonical_property_text(existing) == canonical:
                return replan_history
        return replan_history + (property_text,)

    def _has_replan_seen(self, replan_history: tuple[str, ...], property_text: str):
        canonical = self._canonical_property_text(property_text)
        return any(self._canonical_property_text(existing) == canonical for existing in replan_history)

    def _build_task(self, target, constraints, context_functions, semantic_edges, property_candidate):
        relevant = set(property_candidate.relevant_functions)
        if relevant:
            sel_ctx = [item for item in context_functions if item.full_name in relevant or item.name in relevant][:4]
        elif property_candidate.evidence_type == "cross_function_contract":
            sel_ctx = context_functions[:2]
        else:
            sel_ctx = []
        edge_names = {item.full_name for item in sel_ctx}
        sel_edges = []
        if sel_ctx:
            sel_edges = [
                edge for edge in semantic_edges
                if edge.source in edge_names or edge.target in edge_names or edge.source == target.full_name
            ]
        return TestingTask(target=target, constraints=constraints, property_candidate=property_candidate, context_functions=sel_ctx, semantic_edges=sel_edges)

    def _validator_feedback(self, decision):
        return {
            "error_type": decision.error_type,
            "reasoning": decision.reasoning,
            "fix_suggestion": decision.fix_suggestion,
        }

    def _process_task(self, task: TestingTask, validated_properties: list[str], existing_codes: list[str], replan_history: tuple[str, ...] = ()):
        replan_history = self._extend_replan_history(replan_history, task.property_candidate.property_text)
        code = self.generator.generate_initial_code(task)
        if not code:
            return PropertyAttempt(property_text=task.property_candidate.property_text, status='code_error', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=None, error_message='Failed to produce fenced Python code.')

        attempts = 0
        last_err = ""
        last_reason = ""
        while attempts <= self.max_fix_retries:
            preflight = self.generator.detect_preflight_issues(code)
            if preflight:
                if attempts == self.max_fix_retries:
                    last_err = " ".join(preflight)
                    break
                repaired = self.generator.repair_code(task, code, "Pre-execution harness issue: " + " ".join(preflight))
                if not repaired:
                    last_err = " ".join(preflight)
                    break
                code = repaired
                attempts += 1
                continue

            combined = self.generator.combine_pbt_codes(existing_codes + [code], task)
            retry = self.generator.execute_code(combined)
            if retry["success"]:
                if self.generator.enforce_strength_checks:
                    strength = self.generator.detect_strength_issues(task, code)
                    if strength:
                        if attempts == self.max_fix_retries:
                            return PropertyAttempt(property_text=task.property_candidate.property_text, status='code_error', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code, error_message='Weak test rejected: ' + ' '.join(strength))
                        stronger = self.generator.repair_code(task, code, 'The test passes but is too weak for mutation strength: ' + ' '.join(strength), combined_pbt_code=combined)
                        if not stronger or stronger == code:
                            return PropertyAttempt(property_text=task.property_candidate.property_text, status='code_error', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code, error_message='Weak test rejected: ' + ' '.join(strength))
                        code = stronger
                        attempts += 1
                        continue
                return PropertyAttempt(property_text=task.property_candidate.property_text, status='passed', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code)

            last_err = retry["error_message"] or ""
            decision = self.validator.diagnose(task, code, last_err, combined_pbt_code=combined)
            last_reason = decision.reasoning

            if decision.error_type == "library_defect":
                return PropertyAttempt(property_text=task.property_candidate.property_text, status='library_bug', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code, error_message=last_err, reasoning=decision.reasoning)
            if decision.error_type == "property_defect":
                if len(replan_history) >= self.max_property_replans:
                    return PropertyAttempt(property_text=task.property_candidate.property_text, status='property_error', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code, error_message=last_err, reasoning=f'{decision.reasoning} Property replan budget exhausted after {len(replan_history)} attempts.'.strip())
                repl = self.planner.replan_property(target=task.target, constraints=task.constraints, context_functions=task.context_functions, semantic_edges=task.semantic_edges, existing_properties=sorted(set(validated_properties) | set(replan_history)), original_property=task.property_candidate.property_text, error_reason=decision.reasoning)
                if repl:
                    if self._has_replan_seen(replan_history, repl.property_text):
                        return PropertyAttempt(property_text=task.property_candidate.property_text, status='property_error', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code, error_message=last_err, reasoning=f'{decision.reasoning} Replan loop detected.')
                    return self._process_task(self._build_task(task.target, task.constraints, task.context_functions, task.semantic_edges, repl), validated_properties, existing_codes, replan_history=replan_history + (repl.property_text,))
                return PropertyAttempt(property_text=task.property_candidate.property_text, status='property_error', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code, error_message=last_err, reasoning=decision.reasoning)
            if attempts == self.max_fix_retries:
                break
            repaired = self.generator.repair_code(task, code, last_err, validator_feedback=self._validator_feedback(decision), combined_pbt_code=combined)
            if not repaired:
                break
            code = repaired
            attempts += 1

        return PropertyAttempt(property_text=task.property_candidate.property_text, status='code_error', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code, error_message=last_err, reasoning=last_reason)

    async def _aprocess_task(self, task: TestingTask, validated_properties: list[str], existing_codes: list[str], replan_history: tuple[str, ...] = ()):
        replan_history = self._extend_replan_history(replan_history, task.property_candidate.property_text)
        code = await self.generator.agenerate_initial_code(task)
        if not code:
            return PropertyAttempt(property_text=task.property_candidate.property_text, status='code_error', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=None, error_message='Failed to produce fenced Python code.')

        attempts = 0
        last_err = ""
        last_reason = ""
        while attempts <= self.max_fix_retries:
            preflight = self.generator.detect_preflight_issues(code)
            if preflight:
                if attempts == self.max_fix_retries:
                    last_err = " ".join(preflight)
                    break
                repaired = await self.generator.arepair_code(task, code, 'Pre-execution harness issue: ' + ' '.join(preflight))
                if not repaired:
                    last_err = " ".join(preflight)
                    break
                code = repaired
                attempts += 1
                continue

            combined = self.generator.combine_pbt_codes(existing_codes + [code], task)
            retry = await asyncio.to_thread(self.generator.execute_code, combined)
            if retry["success"]:
                if self.generator.enforce_strength_checks:
                    strength = self.generator.detect_strength_issues(task, code)
                    if strength:
                        if attempts == self.max_fix_retries:
                            return PropertyAttempt(property_text=task.property_candidate.property_text, status='code_error', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code, error_message='Weak test rejected: ' + ' '.join(strength))
                        stronger = await self.generator.arepair_code(task, code, 'The test passes but is too weak for mutation strength: ' + ' '.join(strength), combined_pbt_code=combined)
                        if not stronger or stronger == code:
                            return PropertyAttempt(property_text=task.property_candidate.property_text, status='code_error', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code, error_message='Weak test rejected: ' + ' '.join(strength))
                        code = stronger
                        attempts += 1
                        continue
                return PropertyAttempt(property_text=task.property_candidate.property_text, status='passed', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code)

            last_err = retry["error_message"] or ""
            decision = await self.validator.adiagnose(task, code, last_err, combined_pbt_code=combined)
            last_reason = decision.reasoning

            if decision.error_type == "library_defect":
                return PropertyAttempt(property_text=task.property_candidate.property_text, status='library_bug', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code, error_message=last_err, reasoning=decision.reasoning)
            if decision.error_type == "property_defect":
                if len(replan_history) >= self.max_property_replans:
                    return PropertyAttempt(property_text=task.property_candidate.property_text, status='property_error', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code, error_message=last_err, reasoning=f'{decision.reasoning} Property replan budget exhausted after {len(replan_history)} attempts.'.strip())
                repl = await self.planner.areplan_property(target=task.target, constraints=task.constraints, context_functions=task.context_functions, semantic_edges=task.semantic_edges, existing_properties=sorted(set(validated_properties) | set(replan_history)), original_property=task.property_candidate.property_text, error_reason=decision.reasoning)
                if repl:
                    if self._has_replan_seen(replan_history, repl.property_text):
                        return PropertyAttempt(property_text=task.property_candidate.property_text, status='property_error', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code, error_message=last_err, reasoning=f'{decision.reasoning} Replan loop detected.')
                    return await self._aprocess_task(self._build_task(task.target, task.constraints, task.context_functions, task.semantic_edges, repl), validated_properties, existing_codes, replan_history=replan_history + (repl.property_text,))
                return PropertyAttempt(property_text=task.property_candidate.property_text, status='property_error', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code, error_message=last_err, reasoning=decision.reasoning)
            if attempts == self.max_fix_retries:
                break
            repaired = await self.generator.arepair_code(task, code, last_err, validator_feedback=self._validator_feedback(decision), combined_pbt_code=combined)
            if not repaired:
                break
            code = repaired
            attempts += 1

        return PropertyAttempt(property_text=task.property_candidate.property_text, status='code_error', evidence=task.property_candidate.evidence, evidence_type=task.property_candidate.evidence_type, confidence=task.property_candidate.confidence, mode_label=task.property_candidate.mode_label, oracle_hint=task.property_candidate.oracle_hint, code=code, error_message=last_err, reasoning=last_reason)

    def _execute_counter_implementation(self, task: TestingTask, combined_pbt_code: str, counter_impl: CounterImplementation):
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
                patched = self._patched_test_code(task, combined_pbt_code, counter_impl.code)
                handle.write(patched)
                temp_file = handle.name
            env = os.environ.copy()
            if self.repo_root:
                env["PYTHONPATH"] = f"{self.repo_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
            result = subprocess.run([self.python_executable, '-m', 'pytest', temp_file, '-q', '--tb=short', '-p', 'no:cacheprovider'], capture_output=True, text=True, timeout=self.generator.pytest_timeout_seconds, env=env)
            output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
            return CounterExecutionResult(survived=result.returncode == 0, execution_status='survived' if result.returncode == 0 else 'killed', return_code=result.returncode, output=output, error_message='' if result.returncode == 0 else output)
        except subprocess.TimeoutExpired as exc:
            output = self._normalize_timeout_output(exc.stdout, exc.stderr)
            return CounterExecutionResult(survived=False, execution_status='timeout', return_code=None, output=output, error_message='Counter-implementation execution timed out.')
        except Exception as exc:
            return CounterExecutionResult(survived=False, execution_status='patch_error', return_code=None, output='', error_message=f'{type(exc).__name__}: {exc}')
        finally:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)

    def _normalize_timeout_output(self, stdout: str | bytes | None, stderr: str | bytes | None):
        return f"{self._decode_subprocess_stream(stdout)}\n{self._decode_subprocess_stream(stderr)}".strip()

    def _decode_subprocess_stream(self, value: str | bytes | None):
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    def _recheck_surviving_counterexamples(self, task: TestingTask, combined_pbt_code: str, surviving_counterexamples: list[CounterImplementation], adversarial_checks: list[dict[str, Any]]):
        refreshed: list[CounterImplementation] = []
        first_surv: tuple[CounterImplementation, CounterExecutionResult] | None = None
        for ctr_impl in surviving_counterexamples:
            ctr_exec = self._execute_counter_implementation(task, combined_pbt_code, ctr_impl)
            adversarial_checks.append(self._adversarial_check_record(task=task, counter_implementation=ctr_impl, counter_execution=ctr_exec, source='recheck'))
            if not ctr_exec.survived:
                continue
            refreshed = self._remember_surviving_counterexample(refreshed, ctr_impl)
            if first_surv is None:
                first_surv = (ctr_impl, ctr_exec)
        return refreshed, first_surv

    def _remember_surviving_counterexample(self, counterexamples: list[CounterImplementation], counter_impl: CounterImplementation):
        if any(existing.code == counter_impl.code for existing in counterexamples):
            return counterexamples
        return counterexamples + [counter_impl]

    def _adversarial_check_record(self, task: TestingTask, counter_implementation: CounterImplementation, counter_execution: CounterExecutionResult, source: str):
        return {
            "source": source,
            "property_context": task.property_candidate.property_text,
            "counter_implementation": counter_implementation.to_prompt_dict(),
            "counter_execution": counter_execution.to_dict(),
        }

    def _patched_test_code(self, task: TestingTask, combined_pbt_code: str, counter_impl_code: str):
        target = task.target
        func_name = target.short_name()
        mod_loc = target.module_location
        pcode = self._remove_target_imports(combined_pbt_code, mod_loc, func_name)
        if target.class_name:
            class_name = target.class_name
            patch = f"""
import importlib
target_module = importlib.import_module("{mod_loc}")
target_class = getattr(target_module, "{class_name}")
setattr(target_class, "{func_name}", {func_name})
"""
        else:
            patch = f"""
import importlib
target_module = importlib.import_module("{mod_loc}")
setattr(target_module, "{func_name}", {func_name})
"""
        return f"{counter_impl_code}\n\n{patch}\n\n{pcode}"

    def _remove_target_imports(self, code: str, module_location: str, func_name: str):
        lines_out: list[str] = []
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped.startswith("from "):
                lines_out.append(line)
                continue

            try:
                tree = ast.parse(stripped)
            except SyntaxError:
                lines_out.append(line)
                continue
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ImportFrom):
                lines_out.append(line)
                continue

            import_node = tree.body[0]
            if import_node.module != module_location or import_node.level != 0:
                lines_out.append(line)
                continue

            aliases: list[ast.alias] = []
            bindings: list[str] = []
            removed = False
            for alias in import_node.names:
                if alias.name != func_name:
                    aliases.append(alias)
                    continue
                removed = True
                if alias.asname and alias.asname != func_name:
                    bindings.append(f"{alias.asname} = {func_name}")

            if not removed:
                lines_out.append(line)
                continue

            indent = line[: len(line) - len(line.lstrip())]
            if aliases:
                imports = ', '.join((f'{alias.name} as {alias.asname}' if alias.asname else alias.name for alias in aliases))
                lines_out.append(f"{indent}from {module_location} import {imports}")
            else:
                lines_out.append(f"{indent}# import removed for adversarial patching: {stripped}")
            lines_out.extend(f"{indent}{binding}" for binding in bindings)
        return "\n".join(lines_out)

    def _final_suite_issues(self, target, property_attempts: list[PropertyAttempt], final_code: str):
        passed = [item for item in property_attempts if item.status == "passed"]
        failed = [item for item in property_attempts if item.status != "passed"]
        if not passed:
            return []

        issues: list[str] = []
        narrow = [item for item in passed if self._is_narrow_mode(item)]
        if narrow and len(narrow) == len(passed) and failed:
            issues.append("All accepted properties are narrow boundary/default modes while stronger candidate properties failed.")

        if target.short_name() in {"remove", "discard", "pop"}:
            if all(self._is_exception_only_property(item) for item in passed):
                norm = final_code.lower()
                state_sigs = (
                    "count(",
                    "len(",
                    ".discard(",
                    ".index(",
                    "before =",
                    "after =",
                    "expected",
                    "reference",
                )
                if not any(signal in norm for signal in state_sigs):
                    issues.append("Mutator suite only validates exception behavior on absent inputs and misses successful state-change semantics.")

        return issues

    def _is_narrow_mode(self, attempt: PropertyAttempt):
        text = f"{attempt.mode_label} {attempt.property_text}".lower()
        # TODO: delete
        tokens = (
            "empty iterable",
            "empty input",
            "default-provided",
            "default provided",
            "with default",
            "n is none",
            "full consumption",
            "maxsplit=0",
            "all inputs are scalars",
            "scalar-only",
            "error behavior",
            "must raise",
            "raises",
            "valueerror",
            "non-member",
            "out-of-range",
        )
        return any(token in text for token in tokens)

    def _is_exception_only_property(self, attempt: PropertyAttempt):
        text = f"{attempt.mode_label} {attempt.property_text}".lower()
        tokens = (
            "error behavior",
            "raises",
            "must raise",
            "valueerror",
            "not in list",
            "non-member",
        )
        return any(token in text for token in tokens)


def result_to_jsonl_record(api_info: dict[str, Any], result: ProbeRunResult):
    # gh_info = None
    # if api_info.get("github_url"):
    #     gh_info = {"github_url": api_info.get("github_url"), "commit": api_info.get("tag")}
    # TODO: 把这个变成一个template
    return {
        "api_name": api_info.get("full_name") or f"{api_info.get('module')}.{api_info.get('name')}",
        "package": api_info.get("package"),
        # "library_type": "github" if api_info.get("github_url") else "stdlib",
        # "github_info": gh_info,
        # "api_info": api_info,
        "pbt_code": result.final_pbt_code or "",
        "execution_status": "pass" if result.final_pbt_code else "fail",
        "validation": {
            "status": result.status,
            "strengthening_rounds": result.strengthening_rounds,
            "properties": [item.to_dict() for item in result.properties],
            "bug_report": result.bug_report,
            "adversarial_checks": result.adversarial_checks,
        },
        # "timestamp": datetime.now().isoformat(),
    }
