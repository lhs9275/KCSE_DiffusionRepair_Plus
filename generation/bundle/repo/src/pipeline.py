"""단일 진입점 파이프라인.

사용법:
    python -m src.pipeline --config configs/starcoder2_full.yaml --seed 42 --bugs 1
    python -m src.pipeline --config configs/starcoder2_full.yaml --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from src.utils.seeding import set_all_seeds
from src.replay.input_contract import build_analysis_input, build_generation_input
from src.replay.model_input import validate_model_input_provenance
from src.utils.ast_hints import AST_FREE_POLICY, without_ast_diagnostics
from src.utils.source_coordinates import normalized_bug_line
from src.utils.repair_plans import MAX_PROMPT_EDITS, PLAN_BINDING_POLICY
from src.utils.result_schema import (
    make_bug_result,
    make_candidate_record,
    result_dir,
    save_bug_result,
    require_complete_versioned_generation as _require_complete_versioned_generation,
)
from src.utils.io import load_json, normalize_dict
from src.agents.ast_analyzer import (
    LLMCache,
    analyze_one_bug,
    load_analyzer_model,
    get_code_context,
)
from src.agents.retrieval import RetrievalEngine
from src.agents.diffusion_sampler import load_diffusion_model, sample_plans_for_bug
from src.agents.prompt_generator import create_prompt, build_generation_prompt
from src.agents.patch_generator import (CANDIDATE_POOL_POLICY, CANDIDATE_SELECTION_RANKED,
                                        CANDIDATE_SELECTION_UNRANKED, PatchGeneratorClient,
                                        generate_candidates_for_bug)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# 설정 로드
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    _validate_v2_generation_config(config)
    return config


def _validate_v2_generation_config(config: Dict[str, Any]) -> None:
    protocol = config.get("protocol")
    if protocol not in {"revision_rerun_v2", "revision_rerun_v3"}:
        return
    generation = config.get("generation", {})
    frozen = ({"prompt_count": 1, "k": 10, "max_attempts": 100,
               "batch_size": 8, "temperature": .9, "top_p": .92,
               "repetition_penalty": 1.05, "max_new_tokens": 720}
              if protocol == "revision_rerun_v2" else
              {"prompt_count": 1, "k": 10, "max_attempts": 100,
               "batch_size": 8, "temperature": .8, "top_p": .9,
               "repetition_penalty": 1.05, "max_new_tokens": 640})
    if any(generation.get(key) != value for key, value in frozen.items()):
        raise ValueError(f"{protocol} generation differs from frozen protocol")
    if config.get("training", {}).get("lineage_schema") != "planner_fresh_lineage_v2":
        raise ValueError(f"{protocol} requires fresh planner lineage")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 로그 헤더 기록
# ---------------------------------------------------------------------------

def _write_run_log_header(log_path: Path, config: Dict[str, Any], seed: int, args: argparse.Namespace) -> None:
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== Run Log ===\n")
        f.write(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Config content:\n{json.dumps(config, ensure_ascii=False, indent=2)}\n")
        # GPU 정보
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB\n")
        else:
            f.write("GPU: None (CPU only)\n")
        # 데이터 해시 (SHA256)
        try:
            bugs_meta_path = config["data"]["bugs_meta"]
            result = subprocess.run(
                ["sha256sum", bugs_meta_path], capture_output=True, text=True
            )
            f.write(f"bugs_meta sha256: {result.stdout.strip()}\n")
        except Exception:
            pass
        try:
            ckpt_path = config["checkpoints"]["diffusion_planner"]
            result = subprocess.run(
                ["sha256sum", ckpt_path], capture_output=True, text=True
            )
            f.write(f"checkpoint sha256: {result.stdout.strip()}\n")
        except Exception:
            pass
        f.write("=" * 40 + "\n\n")


def _append_run_log(log_path: Path, message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")


# ---------------------------------------------------------------------------
# 파이프라인 단계별 실행
# ---------------------------------------------------------------------------

def run_pipeline(
    config: Dict[str, Any],
    seed: int,
    results_root: str,
    bug_ids_filter: Optional[List[str]] = None,
    save_every: int = 5,
    compress: bool = False,
) -> None:
    protocol = config.get("protocol")
    v2_primary = protocol in {"revision_rerun_v2", "revision_rerun_v3"}
    if v2_primary:
        _validate_v2_generation_config(config)
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
            raise ValueError("v2 generation requires a real Slurm GPU allocation")
        required_hashes = {
            "bugs_meta": "adc4bbe24a4175ebaa5176c0b8e0dc32376b5aa4aa9e8bb5a75fdfa63c47c5dd",
            "retrieval_corpus": "81011a45926e5dfe6edbb51d013e97a2e43c6c77e8dfe3bd629d81dbdbbb1864",
        }
        for key, expected in required_hashes.items():
            source = Path(config["data"][key])
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(f"v2 primary input {key} differs from frozen SHA-256")
        split = Path(config["training"]["split_manifest"])
        if not split.is_file() or hashlib.sha256(split.read_bytes()).hexdigest() != config["training"]["split_manifest_sha256"]:
            raise ValueError("v2 primary grouped split manifest differs from frozen SHA-256")
        type_map = Path(config["checkpoints"]["type_map"])
        if not type_map.is_file() or hashlib.sha256(type_map.read_bytes()).hexdigest() != "a1ff78928e60a5d69bf1f0c7fc31b10959e49412d6008363c33f03d6426f3d77":
            raise ValueError("v2 primary type map differs from frozen SHA-256")
        for section in ("model", "checkpoints"):
            for value in config[section].values():
                if str(value).startswith("REQUIRES_"):
                    raise ValueError(f"v2 primary {section} asset is not connected: {value}")
        model = config["model"]
        if (model.get("backbone_revision") != "bb9afde76d7945da5745592525db122d4d729eb1" or
                model.get("dense_revision") != "ffb93f3bd4047442299a41ebb6fa998a38507c52"):
            raise ValueError("v2 model revisions differ from the frozen pins")
        model_hashes = {
            "model-00001-of-00003.safetensors": "a8af6f18f9527fece36992ca3cda7ea257f37839539d26f5bf42997edfe3e3f5",
            "model-00002-of-00003.safetensors": "5432ea6c03caf138258efd8acd0434c64a2c22524ed0863671546627d6db6cb0",
            "model-00003-of-00003.safetensors": "da5c636da861b5f0a8e01dbee183e6a814d612c454055449e3a9146d05e0ee3d",
        }
        backbone = Path(model["backbone_path"])
        if Path(model["analyzer_model_path"]).resolve() != backbone.resolve():
            raise ValueError("v2 analyzer and patch backbone must use the same snapshot")
        for name, expected in model_hashes.items():
            path = backbone / name
            if not path.is_file() or _sha256_file(path) != expected:
                raise ValueError(f"v2 StarCoder2 shard {name} differs from frozen SHA-256")
        e5 = Path(model["dense_model_path"]) / "model.safetensors"
        if not e5.is_file() or _sha256_file(e5) != "45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1":
            raise ValueError("v2 E5 weights differ from frozen SHA-256")
        checkpoint = Path(config["checkpoints"]["diffusion_planner"])
        if checkpoint.name != "best_model_state_dict.pt" or not checkpoint.is_file():
            raise ValueError("v2 primary requires a fresh best_model_state_dict.pt")
        provenance_path = checkpoint.parent / "provenance.json"
        if not provenance_path.is_file():
            raise ValueError("v2 primary planner provenance is missing")
        provenance = json.loads(provenance_path.read_text())
        lineage = provenance.get("lineage") or {}
        expected_role = "primary_v2" if protocol == "revision_rerun_v2" else "primary_v3_random"
        if (lineage.get("schema") != "planner_fresh_lineage_v2" or
                lineage.get("kind") != "fresh" or lineage.get("parent_checkpoint") is not None or
                lineage.get("initial_state") != {"epoch": 0, "global_step": 0,
                                                 "attempted_batches": 0, "overflow_skips": 0} or
                provenance.get("run_role") != expected_role or
                provenance.get("execution", {}).get("max_train_batches") is not None or
                provenance.get("execution", {}).get("max_validation_batches") is not None or
                provenance.get("manifest_sha256") != config["training"]["split_manifest_sha256"] or
                provenance.get("config", {}).get("tiny") is not False):
            raise ValueError(f"{protocol} planner is not the frozen fresh full-dimension run")
    cfg_name = config["name"]
    use_ast = config['pipeline'].get('use_ast', True)

    # 결과 디렉토리 / 로그
    res_dir = result_dir(results_root, cfg_name, seed)
    log_path = res_dir / "run.log"

    # --- 이미 완료된 실험 덮어쓰기 방지 ---
    marker = res_dir / ".done"
    if marker.exists():
        logging.warning(f"이미 완료된 실험입니다: {res_dir}. 건너뜁니다.")
        return

    # 데이터 로드
    bugs_meta: Dict[str, Any] = normalize_dict(load_json(config["data"]["bugs_meta"]))
    all_bugs: Dict[str, Any] = normalize_dict(load_json(config["data"]["retrieval_corpus"]))
    if not use_ast:
        all_bugs = {key: without_ast_diagnostics(value) for key, value in all_bugs.items()}

    if bug_ids_filter:
        bugs_meta = {k: v for k, v in bugs_meta.items() if k in bug_ids_filter}

    bug_ids = sorted(bugs_meta.keys(), key=lambda x: int(x) if x.isdigit() else x)
    logging.info(f"처리할 버그 수: {len(bug_ids)} | config={cfg_name} | seed={seed}")

    _write_run_log_header(log_path, config, seed, argparse.Namespace(config=cfg_name))

    # 시드 고정
    set_all_seeds(seed)

    # --- 모델 로딩 ---
    patch_client = PatchGeneratorClient(config["model"]["backbone_path"])
    logging.info("패치 생성 LLM 로드 완료.")

    # AST 분석용 LLM (선택적 — use_ast가 True일 때)
    analyzer_tokenizer = analyzer_model = None
    analyzer_cache: Optional[LLMCache] = None
    if config["pipeline"].get("use_ast", True):
        ast_model_path = config["model"].get("analyzer_model_path") or config["model"]["backbone_path"]
        try:
            analyzer_tokenizer, analyzer_model = load_analyzer_model(ast_model_path)
            cache_path = str(res_dir / "cache" / "llm_reason_cache.json")
            analyzer_cache = LLMCache(cache_path)
            logging.info("AST 분석 LLM 로드 완료.")
        except Exception as e:
            logging.warning(f"AST 분석 LLM 로드 실패 ({e}), AST-only 모드로 진행.")

    # 검색 엔진 (use_retrieval이 True일 때)
    retrieval_engine: Optional[RetrievalEngine] = None
    if config["pipeline"].get("use_retrieval", True):
        dense_model = config["model"].get("dense_model_path")
        cache_dir = str(res_dir / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        retrieval_engine = RetrievalEngine(
            all_bugs=all_bugs,
            bm25_k1=config["retrieval"].get("bm25_k1", 1.6),
            bm25_b=config["retrieval"].get("bm25_b", 0.7),
            corpus_cache=os.path.join(cache_dir, "bm25_corpus_cache.json"),
            dense_model_path=dense_model,
            dense_weight=config["retrieval"].get("dense_weight", 0.6),
            dense_cache=os.path.join(cache_dir, "dense_e5_small_v2.npz"),
            stage_topm=config["retrieval"].get("stage_topm", 200),
        )
        logging.info("검색 엔진 초기화 완료.")

    # 디퓨전 모델
    sampler = type_map = max_nodes = node_type_vocab = None
    if config["pipeline"].get("use_diffusion", True):
        sampler, type_map, max_nodes, node_type_vocab = load_diffusion_model(
            config["checkpoints"]["diffusion_planner"],
            type_map_path=config["checkpoints"].get("type_map"),
        )
        logging.info("디퓨전 모델 로드 완료.")

    # --- 버그별 처리 ---
    saved_count = 0
    failed_bug_ids: List[str] = []
    for bug_idx, bug_id in enumerate(bug_ids, 1):
        t_start = time.time()
        # The answer-bearing benchmark record remains outside every model/cache
        # boundary. AST analysis, retrieval queries, planning and patch generation
        # receive only an explicit buggy-target view.
        item = build_analysis_input(bugs_meta[bug_id])
        _append_run_log(log_path, f"[{bug_idx}/{len(bug_ids)}] bug_id={bug_id} 시작")

        try:
            # 1. AST 분석
            set_all_seeds(seed)  # 각 버그 전 시드 리셋
            if use_ast:
                analyzed = analyze_one_bug(
                    bug_id, item,
                    model=analyzer_model,
                    tokenizer=analyzer_tokenizer,
                    sampling_params=(
                        {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}
                        if analyzer_model else None
                    ),
                    topk=config.get("ast", {}).get("topk", 10),
                    cache=analyzer_cache,
                    enable_rerank=True,
                )
            else:
                analyzed = without_ast_diagnostics(item)
                line = normalized_bug_line(analyzed)
                analyzed['bug_line_in_function'] = line
                analyzed['buggy_line_context'] = get_code_context(
                    (analyzed.get('function') or {}).get('function_before') or '', line, window=3
                ) if line is not None else ''

            # 2. 검색
            retrieved: List[Dict[str, Any]] = []
            retrieval_audit: Dict[str, Any] = {}
            if retrieval_engine is not None:
                retrieved = retrieval_engine.query(
                    func_obj=analyzed,
                    current_bug_id=bug_id,
                    topk=config["retrieval"].get("topk", 5),
                    expand_query=True,
                    audit=retrieval_audit,
                )
                analyzed["bm25"] = {"top": retrieved}

            # 3. 디퓨전 계획
            diffusion_plans: List[Dict[str, Any]] = []
            if sampler is not None:
                diffusion_plans = sample_plans_for_bug(
                    sampler, analyzed, type_map, max_nodes, node_type_vocab,
                    k=config.get("diffusion", {}).get("k", 12),
                    sample_multiplier=config.get("diffusion", {}).get("sample_multiplier", 3),
                    use_ast=use_ast,
                )
                if not diffusion_plans:
                    raise ValueError('repair plan: enabled planner produced no valid plans')

            # 4. 프롬프트 생성. Re-sanitize after upstream stages so neither a
            # model/cache addition nor full-file source crosses the patch boundary.
            generation_input = build_generation_input(
                analyzed,
                use_ast=config["pipeline"].get("use_ast", True),
                use_retrieval=config["pipeline"].get("use_retrieval", True),
            )
            prompt = create_prompt(
                bug_id=bug_id,
                bug_data=generation_input,
                diffusion_candidates=diffusion_plans,
                max_similar=config["retrieval"].get("topk", 5) if config["pipeline"].get("use_retrieval", True) else 0,
                use_ast=config["pipeline"].get("use_ast", True),
                use_retrieval=config["pipeline"].get("use_retrieval", True),
                use_diffusion=config["pipeline"].get("use_diffusion", True),
            )
            edits = diffusion_plans[0].get('edits') if diffusion_plans else []
            prompt = build_generation_prompt(
                (generation_input.get('function') or {}).get('function_before') or '', edits, prompt
            )
            generation_input["prompt"] = prompt

            # 5. 패치 생성
            vram_before = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
            generated = generate_candidates_for_bug(
                client=patch_client,
                bug_id=bug_id,
                base_code=(generation_input.get("function") or {}).get("function_before") or "",
                edits=diffusion_plans[0].get("edits") if diffusion_plans else [],
                prom_row=generation_input,
                target=config.get("generation", {}).get("k", 10),
                batch_size=config.get("generation", {}).get("batch_size", 8),
                temperature=config.get("generation", {}).get("temperature", 0.8),
                top_p=config.get("generation", {}).get("top_p", 0.9),
                repetition_penalty=config.get("generation", {}).get("repetition_penalty", 1.05),
                max_new_tokens=config.get("generation", {}).get("max_new_tokens", 640),
                max_attempts=config.get("generation", {}).get("max_attempts"),
                no_rerank=not config["pipeline"].get("use_ranking", True),
                prepared_prompt=prompt,
                return_accounting=v2_primary,
            )
            if v2_primary:
                patch_candidates, candidate_accounting, valid_unique_raw_pool = generated
            else:
                patch_candidates, candidate_accounting, valid_unique_raw_pool = generated, None, None
            model_input_provenance = patch_client.input_provenance_for(prompt)
            validate_model_input_provenance(
                model_input_provenance, expected_user_prompt=prompt
            )
            vram_peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

            # 결과 직렬화
            bug_total_elapsed = round(time.time() - t_start, 3)
            candidates = []
            for cand in patch_candidates:
                cand_time = cand.get("elapsed_sec")
                candidate_row = make_candidate_record(
                    patch_text=cand.get("code") or "",
                    ast_parse_ok=cand.get("ast", {}).get("ok", False),
                    diff_plan_used=edits[:MAX_PROMPT_EDITS],
                    rank_score=cand.get("score"),
                    tokens_input=cand.get("tokens_in"),
                    tokens_output=cand.get("tokens_out"),
                    inference_time_sec=round(cand_time, 3) if cand_time else bug_total_elapsed,
                )
                if candidate_accounting is not None:
                    candidate_row["temperature"] = cand.get("temperature")
                    candidate_row["top_p"] = cand.get("top_p")
                    candidate_row["plan_ok"] = cand.get("plan_ok")
                    candidate_row["raw_pool_index"] = cand.get("raw_pool_index")
                candidates.append(candidate_row)

            record = make_bug_result(
                bug_id=bug_id,
                seed=seed,
                config=cfg_name,
                candidates=candidates,
                diffusion_plan=diffusion_plans,
                static_analysis_nodes=analyzed.get("suspicious_nodes_topk_reranked"),
                retrieved_examples=retrieved,
                vram_peak_mb=vram_peak,
            )
            record['ast_conditioning'] = 'diagnostic_v1' if use_ast else AST_FREE_POLICY
            record['plan_validation_policy'] = 'source_addressable_v1'
            record['plan_binding_policy'] = PLAN_BINDING_POLICY
            record['candidate_pool_policy'] = CANDIDATE_POOL_POLICY
            record['candidate_selection_policy'] = (
                CANDIDATE_SELECTION_UNRANKED
                if not config['pipeline'].get('use_ranking', True)
                else CANDIDATE_SELECTION_RANKED
            )
            record['source_code_sha256'] = hashlib.sha256(
                ((analyzed.get('function') or {}).get('function_before') or '').encode('utf-8')
            ).hexdigest()
            record['generation_prompt'] = prompt
            record['generation_prompt_sha256'] = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
            record['model_input_provenance'] = model_input_provenance
            if candidate_accounting is not None:
                record['candidate_accounting'] = candidate_accounting
                record['valid_unique_raw_pool'] = valid_unique_raw_pool
            record['retrieval_audit'] = retrieval_audit
            if v2_primary:
                save_bug_result(record, results_root, compress=compress,
                                refuse_existing=True)
            else:
                save_bug_result(record, results_root, compress=compress)
            saved_count += 1

            elapsed = round(time.time() - t_start, 1)
            _append_run_log(log_path, f"bug_id={bug_id} 완료 ({elapsed}s, {len(candidates)}개 후보, VRAM={vram_peak:.0f}MB)")

            if analyzer_cache and saved_count % save_every == 0:
                analyzer_cache.flush()

        except torch.cuda.OutOfMemoryError as e:
            logging.error(f"[{bug_id}] CUDA OOM: {e}")
            torch.cuda.empty_cache()
            _append_run_log(log_path, f"bug_id={bug_id} OOM 오류 — 건너뜀")
            failed_bug_ids.append(bug_id)
        except Exception as e:
            import traceback
            logging.error(f"[{bug_id}] 예외: {e}")
            _append_run_log(log_path, f"bug_id={bug_id} 예외: {e}\n{traceback.format_exc()}")
            failed_bug_ids.append(bug_id)

    if analyzer_cache:
        analyzer_cache.flush()

    # Validate coverage before any success marker (including the empty cohort).
    _require_complete_versioned_generation(
        versioned=v2_primary, saved_count=saved_count,
        expected_count=len(bug_ids), failed_bug_ids=failed_bug_ids,
    )
    # 완료 마커 생성 (최소 1개 이상 저장됐을 때만)
    if (v2_primary and saved_count == len(bug_ids)) or (not v2_primary and saved_count > 0):
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    _append_run_log(log_path, f"=== 실험 완료: {saved_count}/{len(bug_ids)} 버그 처리됨 ===")
    logging.info(f"실험 완료: {saved_count}/{len(bug_ids)} 저장됨 → {res_dir}")


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DiffusionRepair 저널 통합 파이프라인")
    p.add_argument("--config", required=True, help="configs/*.yaml 경로")
    p.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    p.add_argument("--results_root", type=str, default="results", help="결과 저장 루트 디렉토리")
    p.add_argument("--bugs", type=str, default="", help="처리할 bug_id 목록 (쉼표 구분). 비어있으면 전체.")
    p.add_argument("--compress", action="store_true", help="결과 JSON을 gzip으로 압축 저장")
    return p


def main() -> None:
    args = get_parser().parse_args()
    config = load_config(args.config)

    bug_ids_filter: Optional[List[str]] = None
    if args.bugs:
        bug_ids_filter = [b.strip() for b in args.bugs.split(",") if b.strip()]

    run_pipeline(
        config=config,
        seed=args.seed,
        results_root=args.results_root,
        bug_ids_filter=bug_ids_filter,
        compress=args.compress,
    )


if __name__ == "__main__":
    main()
