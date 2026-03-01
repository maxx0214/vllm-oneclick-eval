import os
import json
import time
import argparse
from pathlib import Path

def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"

def dir_size_bytes(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total

def find_model_dir(root: Path) -> Path:
    """
    root 밑에서 가장 그럴듯한 HF 모델 디렉토리 찾기:
      - config.json + (model*.safetensors or *.bin) 존재
    """
    candidates = []
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        cfg = d / "config.json"
        if not cfg.exists():
            continue
        st = list(d.glob("*.safetensors"))
        bn = list(d.glob("pytorch_model*.bin")) + list(d.glob("*.bin"))
        if st or bn:
            candidates.append(d)

    if not candidates:
        raise FileNotFoundError(f"Cannot find model dir under: {root}")

    # 가장 큰 폴더(=가중치 포함 가능성 높음) 우선
    candidates.sort(key=lambda x: dir_size_bytes(x), reverse=True)
    return candidates[0]

def package_check(model_dir: Path) -> dict:
    info = {
        "model_dir": str(model_dir),
        "exists": model_dir.exists(),
        "total_size": human_bytes(dir_size_bytes(model_dir)) if model_dir.exists() else "0B",
        "config_json": (model_dir / "config.json").exists(),
        "generation_config_json": (model_dir / "generation_config.json").exists(),
        "tokenizer_json": (model_dir / "tokenizer.json").exists(),
        "tokenizer_model": (model_dir / "tokenizer.model").exists(),
        "merges_txt": (model_dir / "merges.txt").exists(),
        "vocab_json": (model_dir / "vocab.json").exists(),
        "special_tokens_map_json": (model_dir / "special_tokens_map.json").exists(),
        "chat_template_jinja": (model_dir / "chat_template.jinja").exists(),
        "recipe_yaml": (model_dir / "recipe.yaml").exists(),
        "safetensors_files": [p.name for p in model_dir.glob("*.safetensors")],
        "bin_files": [p.name for p in model_dir.glob("*.bin")],
    }
    info["has_weights"] = (len(info["safetensors_files"]) + len(info["bin_files"])) > 0
    info["looks_like_hf_dir"] = info["config_json"] and info["has_weights"]
    return info

def build_chat_prompt(tokenizer, user_text: str) -> str:
    
    messages = [{"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # ← 핵심: assistant 시작 지점 자동 부착
    )
    return prompt

def run_vllm_smoke_and_bench(
    model_dir: Path,
    gpu_mem_util: float,
    max_model_len: int,
    seed: int,
    enforce_eager: bool,
    fix_mistral_regex: bool,
    out_dir: Path,
) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # tokenizer: mistral regex warning 회피용(가능하면)
    tok_kwargs = {"trust_remote_code": True}
   
    if fix_mistral_regex:
        tok_kwargs["fix_mistral_regex"] = True

    # --- SAFE tokenizer load (transformers 버전마다 fix_mistral_regex 처리 방식이 달라서) ---
    tok_kwargs = dict(tok_kwargs)  # 기존 유지
    fix_flag = tok_kwargs.pop("fix_mistral_regex", None)

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), **tok_kwargs)
    except TypeError as e:
        raise

    
    if fix_flag is True:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), fix_mistral_regex=True, **tok_kwargs)
        except TypeError:
            pass


    
    llm = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        disable_log_stats=True,
        seed=seed,
        enforce_eager=enforce_eager,
    )

    report = {
        "info": {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_dir": str(model_dir),
            "gpu_mem_util": gpu_mem_util,
            "max_model_len": max_model_len,
            "seed": seed,
            "enforce_eager": enforce_eager,
            "fix_mistral_regex": fix_mistral_regex,
        },
        "smoke": [],
        "bench": {},
    }

    # ----------------
    # SMOKE TEST CASES
    # ----------------
    smoke_cases = [
        ("force_digit_50", "다음 규칙을 지켜라.\n규칙: 오직 문자 '1'만 50번 출력하고 다른 말 금지.\n출력:"),
        ("one_sentence", "한 문장으로만 답해. 질문: 파이썬에서 리스트 컴프리헨션이 뭐야?\n답:"),
        ("stop_probe", "답변을 시작할 때 절대로 '<' 문자를 쓰지 마.\n질문: 오늘 기분 어때?\n답:"),
    ]

    sp_smoke = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)

    t0_all = time.perf_counter()
    for name, user_text in smoke_cases:
        prompt = build_chat_prompt(tokenizer, user_text)

        t0 = time.perf_counter()
        out = llm.generate([prompt], sp_smoke)[0].outputs[0]
        t1 = time.perf_counter()

        token_ids = getattr(out, "token_ids", []) or []
        eos_id = getattr(tokenizer, "eos_token_id", None)
        first_token = token_ids[0] if token_ids else None

        # “빈 출력” 의심: 첫 토큰이 EOS이거나, 텍스트가 비어있음
        empty_suspect = (out.text == "") or (eos_id is not None and first_token == eos_id)

        report["smoke"].append({
            "name": name,
            "elapsed_sec": t1 - t0,
            "prompt_preview": user_text[:200],
            "prompt_len_chars": len(prompt),
            "text": out.text,
            "text_len_chars": len(out.text),
            "n_gen_tokens": len(token_ids),
            "finish_reason": out.finish_reason,
            "first_token_id": first_token,
            "eos_token_id": eos_id,
            "empty_output_suspect": empty_suspect,
        })

    t1_all = time.perf_counter()
    report["smoke_total_elapsed_sec"] = t1_all - t0_all

    # ----------------
    # BENCHMARK
    # ----------------
    # 목표: “짧은 입력/중간 입력/긴 입력”에서 decode 성능을 대략 비교
    # - server랑 비슷하게: chat_template prompt
    # - max_model_len 넘지 않게: 입력은 길이 조절, 출력 max_tokens 고정
    def bench_one(label: str, user_text: str, max_out_tokens: int, repeats: int):
        prompt = build_chat_prompt(tokenizer, user_text)

        
        safe_prompt = truncate_to_fit(
            tokenizer,
            prompt,
            max_model_len=max_model_len,
            max_out_tokens=max_out_tokens,
            reserve_tokens=32,
        )
        if safe_prompt != prompt:
            report.setdefault("warnings", []).append({
                "type": "prompt_truncated",
                "label": label,
                "orig_len_chars": len(prompt),
                "new_len_chars": len(safe_prompt),
            })
        prompt = safe_prompt

        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_out_tokens)

        total_tokens = 0
        t0 = time.perf_counter()
        for _ in range(repeats):
            o = llm.generate([prompt], sp)[0].outputs[0]
            token_ids = getattr(o, "token_ids", []) or []
            total_tokens += len(token_ids)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        sec_per_tok = elapsed / max(total_tokens, 1)
        tok_per_sec = (total_tokens / elapsed) if elapsed > 0 else 0.0

        return {
            "prompt_len_chars": len(prompt),
            "repeats": repeats,
            "max_out_tokens": max_out_tokens,
            "total_gen_tokens": total_tokens,
            "total_sec": elapsed,
            "sec_per_token": sec_per_tok,
            "tok_per_sec": tok_per_sec,
        }
    # 입력 길이 만들기 (토크나이즈 길이를 정확히 맞추기 힘드니 “대략적인 길이”로)
    short_in = "한 문장으로만 답해. '안녕'의 영어 표현은?"
    mid_in = "아래 내용을 5줄로 요약해줘:\n" + ("한국어 문장입니다. " * 200)
    long_in = "아래 내용을 핵심만 bullet로 정리:\n" + ("긴 문단 테스트입니다. " * 900)

    report["bench"]["short"] = bench_one("short", short_in, max_out_tokens=256, repeats=3)
    report["bench"]["mid"]   = bench_one("mid",   mid_in,   max_out_tokens=128, repeats=2)
    report["bench"]["long"]  = bench_one("long",  long_in,  max_out_tokens=64,  repeats=1)

    # 저장
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    report["saved"] = str(out_path)
    return report

def truncate_to_fit(tokenizer, prompt: str, max_model_len: int, max_out_tokens: int, reserve_tokens: int = 32) -> str:
    """
    prompt 토큰 길이가 max_model_len을 넘으면, 출력(max_out_tokens) + 여유(reserve_tokens)를 고려해
    입력을 안전하게 자른 뒤 디코딩해서 반환.
    """
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    budget = max_model_len - max_out_tokens - reserve_tokens
    if budget < 64:
        budget = 64
    if len(ids) <= budget:
        return prompt
    ids = ids[-budget:]  # 뒤쪽을 살림(대부분 벤치에서 의미 있음)
    return tokenizer.decode(ids, skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="압축해제 루트(예: .../__unzip_test). 내부에서 model_dir 자동 탐지")
    ap.add_argument("--model", type=str, default=None, help="모델 폴더 직접 지정(예: .../__unzip_test/model)")
    ap.add_argument("--gpu-mem", type=float, default=float(os.environ.get("GPU_MEM_UTIL", "0.70")))
    ap.add_argument("--max-len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "4096")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    ap.add_argument("--enforce-eager", action="store_true", help="torch.compile/cudagraph 최적화 영향 줄이려면 켜기(서버 eager일 때 유사)")
    ap.add_argument("--no-fix-mistral-regex", action="store_true")
    args = ap.parse_args()

    if args.model is None and args.root is None:
        raise SystemExit("Provide --root or --model")

    if args.model:
        model_dir = Path(args.model).expanduser().resolve()
    else:
        root = Path(args.root).expanduser().resolve()
        model_dir = find_model_dir(root)

    print(f"[INFO] model_dir={model_dir}")
    print(f"[INFO] gpu_mem_util={args.gpu_mem}, max_model_len={args.max_len}, seed={args.seed}, enforce_eager={args.enforce_eager}")

    pkg = package_check(model_dir)
    print("\n[CHECK] package summary")
    for k in [
        "exists","total_size","config_json","generation_config_json","tokenizer_json",
        "merges_txt","vocab_json","special_tokens_map_json","chat_template_jinja",
        "recipe_yaml","has_weights","looks_like_hf_dir"
    ]:
        print(f"  - {k}: {pkg.get(k)}")
    print(f"  - safetensors_files: {pkg.get('safetensors_files')}")
    print(f"  - bin_files: {pkg.get('bin_files')}")

    if not pkg["looks_like_hf_dir"]:
        raise SystemExit("[FATAL] This directory doesn't look like a valid HF model folder (missing config/weights).")

    out_dir = model_dir.parent / "reports"
    fix_mistral_regex = not args.no_fix_mistral_regex

    print(f"\n[RUN] vLLM smoke + bench (report -> {out_dir}/eval_report.json)")
    rep = run_vllm_smoke_and_bench(
        model_dir=model_dir,
        gpu_mem_util=args.gpu_mem,
        max_model_len=args.max_len,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        fix_mistral_regex=fix_mistral_regex,
        out_dir=out_dir,
    )

    # 콘솔 요약
    print("\n[SMOKE] results")
    for r in rep["smoke"]:
        flag = "EMPTY?" if r["empty_output_suspect"] else "OK"
        print(f"  - {r['name']}: {flag}, reason={r['finish_reason']}, tokens={r['n_gen_tokens']}, first={r['first_token_id']}")

    print("\n[BENCH] results")
    for k, v in rep["bench"].items():
        print(f"  - {k}: tok/s={v['tok_per_sec']:.2f}, sec/tok={v['sec_per_token']:.6f}, total_tokens={v['total_gen_tokens']}, total_sec={v['total_sec']:.3f}")

    print(f"\n[SAVED] {rep['saved']}")
    print("[DONE]")

if __name__ == "__main__":
    main()
