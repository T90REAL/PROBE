from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe.llm import LLMClient, RuntimeConfig
from probe.repository import RepositoryIndex, find_import_root, infer_package_root
from probe.runner import ProbeRunner


def main():
    parser = argparse.ArgumentParser(description="Reliable Software Testing")
    parser.add_argument("--target-function", required=True, help="Function name, e.g. SeqBase.coeff or module.Class.method")
    parser.add_argument("--package", required=True, help="Package name")
    parser.add_argument("--module", help="Module path for the target")
    parser.add_argument("--file-path", help="Optional source file path for inferring package root")
    parser.add_argument("--config", help="Path to TOML runtime config")
    parser.add_argument("--api-key", help="Override API key without editing config")
    parser.add_argument("--base-url", help="Override OpenAI-compatible base URL without editing config")
    parser.add_argument("--max-refinement-rounds", type=int, default=3)
    parser.add_argument("--max-fix-retries", type=int, default=3)
    parser.add_argument("--max-initial-properties", type=int, default=3)
    args = parser.parse_args()

    config = RuntimeConfig.load(args.config, api_key_override=args.api_key, base_url_override=args.base_url)
    llm = LLMClient(config)

    if args.file_path:
        pkg_root = infer_package_root(args.file_path, args.package)
        repo_root = pkg_root.parent
    else:
        pkg_root, repo_root = find_import_root(args.package)

    index = RepositoryIndex.from_package(pkg_root, args.package)
    tgt_name = args.target_function
    if args.module and tgt_name.startswith(f"{args.module}."):
        tgt_name = tgt_name[len(args.module) + 1:]
    api_info = {
        "name": ".".join(tgt_name.split(".")[-2:]) if tgt_name.count(".") >= 1 else tgt_name,
        "full_name": f"{args.module}.{tgt_name}" if args.module else args.target_function,
        "module": args.module or "",
        "package": args.package,
        "api_type": "method" if "." in tgt_name else "function",
        "file_path": args.file_path or "",
    }

    runner = ProbeRunner(llm=llm, python_executable=config.python_executable, repo_root=str(repo_root), max_fix_retries=args.max_fix_retries, max_refinement_rounds=args.max_refinement_rounds, max_initial_properties=args.max_initial_properties, enforce_strength_checks=config.enforce_strength_checks, reject_weak_suites=config.reject_weak_suites, pytest_timeout_seconds=config.pytest_timeout_seconds)
    result = runner.run(api_info, index)
    print(json.dumps({'status': result.status, 'target_function': result.target_function, 'strengthening_rounds': result.strengthening_rounds, 'final_pbt_code': result.final_pbt_code, 'properties': [item.to_dict() for item in result.properties], 'bug_report': result.bug_report, 'adversarial_checks': result.adversarial_checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
