"""One-case Full/BASE integration with reference method behavior and read-only evidence."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path('/output')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def source_nodes(code, maximum):
    return {i for i, node in enumerate(ast.walk(ast.parse(code)))
            if i < maximum and getattr(node, 'lineno', None) is not None}


def neutralize_base_plan_language(prompt):
    """Remove only generic plan instructions from the no-plan BASE condition."""
    replacements = {
        "Context (use if helpful; prioritize any 'Recommended Repair Plan'):":
            "Context (use if helpful):",
        "- Follow the 'Recommended Repair Plan' when provided; ignore it only if it conflicts with the code/tests.\n":
            "",
        "- If a 'Recommended Repair Plan' is present above, you MUST follow it.\n": "",
        "- Modify ONLY lines covered by the repair plan.\n": "",
    }
    result = prompt
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def non_addressable_returned_edit_nodes(nodes, valid_nodes):
    """Check only edit nodes returned by the reference sampler (KEEP is already removed)."""
    return [node for node in nodes if node not in valid_nodes]


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('real Slurm allocation required')
    arm = os.environ.get('KCSE_ARM')
    if arm not in {'full', 'base'}:
        raise ValueError('KCSE_ARM must be full or base')
    bug_id = os.environ.get('KCSE_BUG_ID', '2')
    seed = int(os.environ.get('KCSE_SEED', '42'))

    import torch
    from src.agents import ast_analyzer as aa
    from src.agents import patch_generator as pg
    from src.agents.diffusion_sampler import load_diffusion_model, sample_plans_for_bug
    from src.agents.patch_generator import (
        PatchGeneratorClient, generate_candidates_for_bug, extract_correct_block,
        ast_ok, validate_candidate_code, canon, _plan_compliance,
        _extract_repair_plan_hints,
    )
    from src.agents.prompt_generator import create_prompt
    from src.agents.retrieval import RetrievalEngine
    from src.replay.input_contract import build_analysis_input, build_generation_input, validate_base_contract
    from src.utils.seeding import set_all_seeds

    expected = {
        '/bundle/repo/src/agents/ast_analyzer.py': '19859d7c821271da87ea22c6af50ca0b5de685d69f18009e902bda882c94c531',
        '/bundle/repo/src/agents/diffusion_sampler.py': 'c3b579e37cfd9536f83edae840d2bca983dcc5311ee7f2e9e9f09bb1a79f04aa',
        '/bundle/repo/src/agents/patch_generator.py': '1ee0ea6cba69298efc82235a5e63543a92438c8c9d4b1d356fbc88ef0c64f2fe',
        '/bundle/repo/src/agents/prompt_generator.py': 'c7f86ebba0ccce1aa829ff8541b557c14a38aab479d288e40696f838e05f890c',
        '/bundle/repo/src/agents/retrieval.py': '63d1a9199a8f59af0ea15480a32f85905f8c506f56a1c7d6bc22f186118d5039',
        '/bundle/assets/metadata.json': 'adc4bbe24a4175ebaa5176c0b8e0dc32376b5aa4aa9e8bb5a75fdfa63c47c5dd',
        '/bundle/assets/corpus.json': '81011a45926e5dfe6edbb51d013e97a2e43c6c77e8dfe3bd629d81dbdbbb1864',
        '/bundle/assets/type_map.json': 'a1ff78928e60a5d69bf1f0c7fc31b10959e49412d6008363c33f03d6426f3d77',
        '/planner/best_model_state_dict.pt': 'b02f95aa1fcb24b19f5a5e038e47976125ce7979a686821810465c3f4ee2f730',
        '/models/e5-small-v2/model.safetensors': '45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1',
    }
    model_manifest = json.loads(Path('/bundle/assets/model-manifest.json').read_text())
    for name, digest in model_manifest.items():
        expected[f'/models/starcoder2-7b/{name}'] = digest
    for path, digest in expected.items():
        if sha(path) != digest:
            raise RuntimeError(f'asset mismatch: {path}')
    dump('verified-assets.json', expected)

    metadata = json.loads(Path('/bundle/assets/metadata.json').read_text())
    analysis = build_analysis_input(metadata[bug_id])
    stages = []
    def stage(name, **fields):
        stages.append({'stage': name, 'time': time.time(), **fields})
        dump('stages.json', stages)

    set_all_seeds(seed)
    stage('patch_load_start')
    patch = PatchGeneratorClient('/models/starcoder2-7b')
    stage('patch_load_complete', gpu_bytes=torch.cuda.memory_allocated())

    model_inputs, raw_generations = [], []
    original_generate = patch.generate
    def observed_generate(prompts, args):
        chats = []
        modes = []
        for prompt in prompts:
            messages = ([{'role': 'system', 'content': patch.system_prompt}]
                        if patch.system_prompt else []) + [{'role': 'user', 'content': prompt}]
            try:
                chats.append(patch.tokenizer.apply_chat_template(messages, tokenize=False,
                                                                  add_generation_prompt=True))
                modes.append('chat_template')
            except Exception:
                chats.append(prompt)
                modes.append('legacy_plain_fallback')
        full = patch.tokenizer(chats, add_special_tokens=True, truncation=False)
        used = patch.tokenizer(chats, add_special_tokens=True, truncation=True,
                               max_length=args.get('max_input_tokens', 4096))
        for prompt, chat, mode, before, after in zip(prompts, chats, modes,
                                                     full['input_ids'], used['input_ids']):
            model_inputs.append({
                'user_prompt': prompt, 'rendered_input': chat, 'render_mode': mode,
                'user_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'rendered_sha256': hashlib.sha256(chat.encode()).hexdigest(),
                'untruncated_tokens': len(before), 'used_tokens': len(after),
                'truncated': len(before) != len(after), 'truncation_side': patch.tokenizer.truncation_side,
                'used_token_ids': after,
            })
        outputs = original_generate(prompts, args)
        for prompt, output in zip(prompts, outputs):
            raw_generations.append({'call_args': dict(args),
                                    'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                                    'output': output})
        return outputs
    patch.generate = observed_generate

    ast_evidence = []
    retrieval_rows = []
    sampler_rows = []
    plans = []
    retrieved = []

    if arm == 'full':
        stage('analyzer_load_start')
        tokenizer, analyzer = aa.load_analyzer_model('/models/starcoder2-7b')
        stage('analyzer_load_complete', gpu_bytes=torch.cuda.memory_allocated())
        original_query = aa.query_llm_batch
        def audited_query(tokenizer, model, messages_list, sampling_params,
                          max_new_tokens=192, timeout_s=30, retry=1, batch_size=8):
            # Preserve the reference function's result and RNG behavior; record its exact inputs/results.
            texts = []
            for messages in messages_list:
                if getattr(tokenizer, 'chat_template', None):
                    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                         add_generation_prompt=True)
                else:
                    parts = []
                    for message in messages:
                        parts.append(f"<|{message.get('role','')}|>\n{message.get('content','')}")
                    parts.append('<|assistant|>')
                    text = '\n'.join(parts)
                texts.append(text)
            before = tokenizer(texts, add_special_tokens=True, truncation=False)['input_ids']
            results = original_query(tokenizer, model, messages_list, sampling_params,
                                     max_new_tokens=max_new_tokens, timeout_s=timeout_s,
                                     retry=retry, batch_size=batch_size)
            for index, (messages, text, tokens, result) in enumerate(zip(messages_list, texts, before, results)):
                ast_evidence.append({'query_index': index, 'messages': messages,
                                     'rendered_input': text, 'input_token_ids': tokens,
                                     'result': result})
            return results
        aa.query_llm_batch = audited_query
        stage('ast_start')
        analyzed = aa.analyze_one_bug(
            bug_id, analysis, model=analyzer, tokenizer=tokenizer,
            sampling_params={'temperature': .7, 'top_p': .8, 'top_k': 20,
                             'repetition_penalty': 1.05},
            topk=10, cache=aa.LLMCache(), enable_rerank=True,
        )
        dump('analyzed.json', analyzed)
        dump('ast-evidence.json', ast_evidence)
        stage('ast_complete', nodes=len(analyzed.get('suspicious_nodes_topk') or []))

        corpus = json.loads(Path('/bundle/assets/corpus.json').read_text())
        retrieval = RetrievalEngine(all_bugs=corpus, bm25_k1=1.6, bm25_b=.7,
                                    corpus_cache=str(OUT/'corpus-cache.json'),
                                    dense_model_path='/models/e5-small-v2', dense_weight=.6,
                                    dense_cache=str(OUT/'dense-cache.npz'), stage_topm=200)
        retrieved = retrieval.query(analyzed, bug_id, topk=5, expand_query=True)
        target = ast.dump(ast.parse(analysis['function']['function_before']), include_attributes=False)
        for row in retrieved:
            function = row.get('function') or {}
            matches = []
            for field in ('function_before', 'function_after'):
                try:
                    matches.append(ast.dump(ast.parse(function.get(field) or ''),
                                            include_attributes=False) == target)
                except SyntaxError:
                    matches.append(False)
            retrieval_rows.append({'id': row.get('id'), 'score': row.get('score_reranked'),
                                   'target_matches_before': matches[0],
                                   'target_matches_after': matches[1]})
        analyzed['bm25'] = {'top': retrieved}
        dump('retrieval.json', {'results': retrieved, 'relations': retrieval_rows})
        stage('retrieval_complete', rows=len(retrieved))

        sampler, type_map, max_nodes, vocab = load_diffusion_model(
            '/planner/best_model_state_dict.pt', '/bundle/assets/type_map.json')
        valid = source_nodes(analysis['function']['function_before'], max_nodes)
        original_sample = sampler.sample
        def observed_sample(*args, **kwargs):
            ops, nodes = original_sample(*args, **kwargs)
            # The reference sampler already removes KEEP entries before returning.
            # Do not reject an irrelevant KEEP node that was present internally.
            row = {'ops': ops, 'nodes': nodes,
                   'non_addressable_returned_edit_nodes':
                       non_addressable_returned_edit_nodes(nodes, valid)}
            sampler_rows.append(row)
            if row['non_addressable_returned_edit_nodes']:
                raise RuntimeError('sampled non-KEEP edit is not source-addressable')
            return ops, nodes
        sampler.sample = observed_sample
        plans = sample_plans_for_bug(sampler, analyzed, type_map, max_nodes, vocab,
                                     k=12, sample_multiplier=3)
        dump('sampler-evidence.json', {'draws': sampler_rows, 'plans': plans})
        stage('planner_complete', draws=len(sampler_rows), plans=len(plans))
        generation_input = build_generation_input(analyzed, use_ast=True, use_retrieval=True)
        prompt = create_prompt(bug_id, generation_input, plans, max_similar=5,
                               use_ast=True, use_retrieval=True, use_diffusion=True)
        edits = (plans[0].get('edits') or []) if plans else []
    else:
        generation_input = build_generation_input(analysis, use_ast=False, use_retrieval=False)
        prompt = create_prompt(bug_id, generation_input, [], max_similar=5,
                               use_ast=False, use_retrieval=False, use_diffusion=False)
        prompt = neutralize_base_plan_language(prompt)
        edits = []
        base_check = validate_base_contract(prompt=prompt, edits=edits, plans=[],
                                            generation_input=generation_input)
        dump('base-contract.json', base_check)
        stage('base_input_complete')

        original_guided_prompt = pg._build_guided_prompt
        pg._build_guided_prompt = lambda base_code, sent_edits, full_prompt: (
            neutralize_base_plan_language(
                original_guided_prompt(base_code, sent_edits, full_prompt)
            )
        )

    generation_input['prompt'] = prompt
    dump('pre-generator-input.json', {'arm': arm, 'generation_input': generation_input,
                                      'plans': plans, 'edits': edits})
    stage('generation_start')
    candidates = generate_candidates_for_bug(
        patch, bug_id, analysis['function']['function_before'], edits, generation_input,
        target=10, batch_size=8, temperature=.8, top_p=.9,
        repetition_penalty=1.05, max_new_tokens=640, no_rerank=False)
    if arm == 'base':
        final_base_checks = [
            validate_base_contract(prompt=row['user_prompt'], edits=[], plans=[],
                                   generation_input=generation_input)
            for row in model_inputs
        ]
        dump('base-final-model-input-contract.json', final_base_checks)

    # Reconstruct reference admission over every observed raw output for an auditable reason ledger.
    expected_name = (analysis.get('function') or {}).get('function_name') or ''
    hints = _extract_repair_plan_hints(model_inputs[0]['user_prompt']) if model_inputs else []
    seen = set()
    raw_audit = []
    for index, row in enumerate(raw_generations):
        text = row['output'].get('text') or ''
        code = extract_correct_block(text)
        reason = None
        ok = False
        plan_ok = None
        if not code.strip():
            reason = 'empty_correct_block'
        else:
            ok, error = ast_ok(code)
            if not ok:
                reason = 'ast_parse:' + str(error)
            else:
                valid_code, validation = validate_candidate_code(
                    code, analysis['function']['function_before'], expected_name)
                digest = hashlib.sha1(canon(code).encode()).hexdigest()
                if not valid_code:
                    reason = 'candidate_validation:' + validation
                elif digest in seen:
                    reason = 'duplicate'
                else:
                    seen.add(digest)
                    plan_ok = _plan_compliance(model_inputs[0]['user_prompt'],
                                               analysis['function']['function_before'],
                                               code, hints)
                    reason = 'admitted_compliant' if plan_ok or not hints else 'admitted_fallback'
        try:
            compile(ast.parse(code), '<candidate-diagnostic>', 'exec')
            compile_result = {'ok': True, 'error': None}
        except Exception as error:
            compile_result = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
        raw_audit.append({'index': index, 'reason': reason, 'plan_ok': plan_ok,
                          'compile_diagnostic': compile_result,
                          'code_sha256': hashlib.sha256(code.encode()).hexdigest() if code else None,
                          'canonical_sha1': hashlib.sha1(canon(code).encode()).hexdigest() if code else None})
    final = []
    for index, candidate in enumerate(candidates):
        code = candidate.get('code') or ''
        try:
            compile(ast.parse(code), '<candidate-diagnostic>', 'exec')
            compile_result = {'ok': True, 'error': None}
        except Exception as error:
            compile_result = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
        final.append({'index': index, 'patch_text': code,
                      'patch_sha256': hashlib.sha256(code.encode()).hexdigest(),
                      'rank_score': candidate.get('score'), 'plan_ok': candidate.get('plan_ok'),
                      'compile_diagnostic': compile_result,
                      'temperature': candidate.get('temperature'), 'top_p': candidate.get('top_p')})
    raw_canonical = {row['canonical_sha1'] for row in raw_audit if row['canonical_sha1']}
    final_canonical = [hashlib.sha1(canon(row['patch_text']).encode()).hexdigest()
                       for row in final]
    missing_from_raw = [digest for digest in final_canonical if digest not in raw_canonical]
    if missing_from_raw:
        raise RuntimeError(f'final candidates missing from raw-generation ledger: {missing_from_raw}')
    dump('model-inputs.json', model_inputs)
    dump('raw-generations.json', raw_generations)
    dump('candidate-audit.json', raw_audit)
    dump('candidate-linkage.json', {'final_canonical_sha1_in_rank_order': final_canonical,
                                    'missing_from_raw': missing_from_raw})
    dump('result.json', {'bug_id': bug_id, 'seed': seed, 'arm': arm,
                         'candidates': final, 'candidate_count': len(final),
                         'retrieved_examples': retrieved, 'diffusion_plan': plans,
                         'static_analysis_nodes': (analyzed.get('suspicious_nodes_topk_reranked') or [])
                         if arm == 'full' else []})
    stage('generation_complete', candidates=len(final), raw_outputs=len(raw_generations))
    dump('GENERATION_COMPLETED.json', {'status': 'GENERATED_NOT_EVALUATED', 'arm': arm,
                                       'bug_id': bug_id, 'seed': seed,
                                       'candidate_count': len(final),
                                       'result_sha256': sha(OUT/'result.json'),
                                       'job_id': os.environ['SLURM_JOB_ID']})


if __name__ == '__main__':
    main()
