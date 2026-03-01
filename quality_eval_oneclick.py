#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# vLLM
from vllm import LLM, SamplingParams

# HF tokenizer (프롬프트 길이 안전장치/토큰수 측정용)
from transformers import AutoTokenizer


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_tokenizer(model_dir: Path):
    """
    Exaone 계열/미스트랄 계열 경고(incorrect regex pattern)가 떠도 실행은 됨.
    다만 Transformers 버전에 따라 fix_mistral_regex 인자 충돌이 있을 수 있어서
    여기서는 "안전하게" 그냥 기본 로딩으로 가고, 실패하면 fallback.
    """
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        return tok
    except TypeError as e:
        # 특정 버전에서 fix_mistral_regex 중복 전달 등 이슈가 생길 수 있음
        # 가장 보수적으로 재시도
        tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
        return tok


def build_prompt(tokenizer, user_text: str) -> str:
    """
    서버가 chat template 기반일 가능성이 크므로 apply_chat_template 우선.
    없으면 plain으로 fallback.
    """
    msgs = [{"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            pass
    # fallback
    return user_text


def truncate_to_fit(tokenizer, prompt: str, max_model_len: int, max_out_tokens: int) -> str:
    """
    vLLM에서 'prompt length > max_model_len' 에러 방지.
    남길 토큰 수 = max_model_len - max_out_tokens - 약간의 여유
    """
    budget = max_model_len - max_out_tokens - 8
    if budget < 64:
        budget = max(16, max_model_len // 4)

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= budget:
        return prompt

    ids2 = ids[-budget:]  # 뒤쪽 유지(대화 템플릿이면 뒤가 중요)
    return tokenizer.decode(ids2, skip_special_tokens=False)


def safe_generate(llm: LLM, tokenizer, user_text: str, max_model_len: int, max_out_tokens: int,
                  temperature: float = 0.0, top_p: float = 1.0, stop: List[str] = None) -> Dict[str, Any]:
    prompt = build_prompt(tokenizer, user_text)
    prompt2 = truncate_to_fit(tokenizer, prompt, max_model_len=max_model_len, max_out_tokens=max_out_tokens)

    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_out_tokens,
        stop=stop
    )

    t0 = time.perf_counter()
    out = llm.generate([prompt2], sp)[0].outputs[0]
    t1 = time.perf_counter()

    text = out.text or ""
    token_ids = getattr(out, "token_ids", []) or []
    finish_reason = getattr(out, "finish_reason", None)

    first_token_id = token_ids[0] if token_ids else None
    eos_id = getattr(tokenizer, "eos_token_id", None)

    empty_suspect = (len(text.strip()) == 0) or (first_token_id == eos_id)

    return {
        "user_text": user_text[:200],
        "prompt_len_chars": len(prompt2),
        "elapsed_sec": (t1 - t0),
        "text": text,
        "text_len_chars": len(text),
        "n_gen_tokens": len(token_ids),
        "finish_reason": finish_reason,
        "first_token_id": first_token_id,
        "eos_token_id": eos_id,
        "prompt_truncated": (prompt2 != prompt),
        "empty_output_suspect": empty_suspect,
    }


# -------------------------
# 평가 케이스 정의
# -------------------------

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def extract_first_json(text: str) -> Tuple[bool, Any]:
    """
    출력에서 첫 JSON object를 찾아 파싱 시도.
    """
    t = text.strip()
    # 단순하게 첫 '{' ~ 마지막 '}' 범위를 잡아봄
    if "{" not in t or "}" not in t:
        return False, None
    start = t.find("{")
    end = t.rfind("}")
    cand = t[start:end+1]
    try:
        return True, json.loads(cand)
    except Exception:
        return False, None


def check_exact(text: str, expected: str) -> bool:
    return norm(text) == norm(expected)


def check_only_char(text: str, ch: str, count: int) -> bool:
    t = text.strip()
    return (len(t) == count) and all(c == ch for c in t)


def check_int_answer(text: str, expected_int: int) -> bool:
    # 출력에서 첫 정수 하나만 뽑아 비교
    m = re.search(r"-?\d+", text)
    if not m:
        return False
    return int(m.group(0)) == expected_int


def check_one_sentence(text: str) -> bool:
    # 문장 종결부호 기준으로 너무 많으면 실패(대충)
    t = text.strip()
    if len(t) == 0:
        return False
    # 줄바꿈이 여러개면 실패
    if t.count("\n") >= 1:
        return False
    # 마침표/물음표/느낌표 기준 2개 이상이면 실패로 간주(완벽하진 않지만 지표로 쓸만)
    enders = sum(t.count(x) for x in [".", "?", "!", "。", "！", "？"])
    return enders <= 1


def build_cases() -> List[Dict[str, Any]]:
    """
    기존 7문항과 "동일한 유형/채점방식"으로 확장 버전.
    - format: 반복문자 / JSON만 출력 / 한 문장 / 금지문자
    - accuracy: 간단 산수(숫자만 요구) -> check_int_answer 사용(기존과 동일)
    - sanity: 모르면 모른다고 한 문장(기존과 동일)
    """
    cases: List[Dict[str, Any]] = []

    # -------------------------
    # 1) 반복 문자(only_1x50) 유형 확장 (format)
    # -------------------------
    repeat_specs = [
        ("only_1x50", "1", 50, 80),
        ("only_0x50", "0", 50, 80),
        ("only_1x32", "1", 32, 64),
        ("only_2x40", "2", 40, 80),
        ("only_ax60", "a", 60, 96),
        ("only_Ax60", "A", 60, 96),
        ("only_Zx30", "Z", 30, 64),
        ("only_star_x40", "*", 40, 96),
        ("only_hash_x40", "#", 40, 96),
        ("only_kor_x20", "가", 20, 64),
    ]
    for name, ch, cnt, max_out in repeat_specs:
        cases.append({
            "name": name,
            "user": f"오직 문자 '{ch}'만 {cnt}번 출력해. 다른 글자, 공백, 줄바꿈 금지. 출력만 해.",
            "max_out": max_out,
            "grade": (lambda out, ch=ch, cnt=cnt: check_only_char(out, ch, cnt)),
            "type": "format",
        })

    # -------------------------
    # 2) 간단 산수(정답 고정) 유형 확장 (accuracy)
    #    - 기존과 똑같이 "정답 숫자만 출력" 문구 + check_int_answer 사용
    # -------------------------
    math_specs = [
        ("math_3_9_add", "3+9는? 정답 숫자만 출력해.", 12),
        ("math_88_12_sub", "88-12는? 정답 숫자만 출력해.", 76),
        ("math_12_34_mul", "12*34는? 정답 숫자만 출력해.", 408),
        ("math_81_9_div", "81/9는? 정답 숫자만 출력해.", 9),
        ("math_999_1_add", "999+1은? 정답 숫자만 출력해.", 1000),
        ("math_50_8_mod", "50을 8로 나눈 나머지는? 정답 숫자만 출력해.", 2),
        ("math_neg_17_5_add", "-17+5는? 정답 숫자만 출력해.", -12),
        ("math_7_7_mul", "7*7은? 정답 숫자만 출력해.", 49),
        ("math_144_sqrt", "144의 제곱근(양의 정수)은? 정답 숫자만 출력해.", 12),
        ("math_2_pow_10", "2의 10제곱은? 정답 숫자만 출력해.", 1024),
        ("math_365_days", "윤년이 아닌 해는 며칠이야? 정답 숫자만 출력해.", 365),
        ("math_123_7", "123*7은? 정답 숫자만 출력해.", 861),  # 기존과 동일 케이스 포함
        ("math_17_25", "17+25는? 정답 숫자만 출력해.", 42),    # 기존과 동일 케이스 포함
    ]
    for name, user, ans in math_specs:
        cases.append({
            "name": name,
            "user": user,
            "max_out": 16,
            "grade": (lambda out, ans=ans: check_int_answer(out, ans)),
            "type": "accuracy",
        })

    # -------------------------
    # 3) JSON만 출력 유형 확장 (format)
    #    - 기존과 동일하게 "extract_first_json == expected" 로 채점
    #    - (코드블럭 허용/불허는 기존처럼 '허용' 상태)
    # -------------------------
    json_specs = [
        ("json_ab_1_2", {"a": 1, "b": 2}),
        ("json_xy_10_20", {"x": 10, "y": 20}),
        ("json_user_age", {"user": "kim", "age": 20}),
        ("json_flag", {"ok": True, "n": 3}),
        ("json_nested", {"a": {"b": 2}, "c": [1, 2]}),
        ("json_list", {"items": [3, 1, 2]}),
    ]
    for name, obj in json_specs:
        # JSON 문자열을 stable하게 만들기 (공백/키순서 영향 줄이기)
        json_str = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        cases.append({
            "name": name,
            "user": f"다음 JSON만 출력해: {json_str}. 다른 말 절대 금지.",
            "max_out": 64,
            "grade": (lambda out, obj=obj: (extract_first_json(out)[0] and extract_first_json(out)[1] == obj)),
            "type": "format",
        })

    # -------------------------
    # 4) 한 문장 강제 유형 확장 (format)
    #    - 기존과 동일하게 check_one_sentence
    # -------------------------
    one_sentence_prompts = [
        ("one_sentence_define_listcomp", "한 문장으로만 답해. 질문: 파이썬 리스트 컴프리헨션이 뭐야?"),
        ("one_sentence_define_dict", "한 문장으로만 답해. 질문: 파이썬 딕셔너리가 뭐야?"),
        ("one_sentence_define_numpy", "한 문장으로만 답해. 질문: NumPy가 뭐야?"),
        ("one_sentence_define_git", "한 문장으로만 답해. 질문: Git이 뭐야?"),
        ("one_sentence_define_overfit", "한 문장으로만 답해. 질문: 오버피팅이 뭐야?"),
        ("one_sentence_define_quant", "한 문장으로만 답해. 질문: 양자화(quantization)가 뭐야?"),
        ("one_sentence_define_vllm", "한 문장으로만 답해. 질문: vLLM이 뭐야?"),
        ("one_sentence_define_transformer", "한 문장으로만 답해. 질문: 트랜스포머가 뭐야?"),
    ]
    for name, user in one_sentence_prompts:
        cases.append({
            "name": name,
            "user": user,
            "max_out": 64,
            "grade": (lambda out: check_one_sentence(out)),
            "type": "format",
        })

    # -------------------------
    # 5) 금지 문자(각종 변형) (format)
    #    - 기존 no_angle_bracket 로직: "<"가 앞 20자에 없고, 한 문장이면 pass
    #    - 동일 스타일로 다른 금지문자도 추가(앞 20자 검사 + 한 문장)
    # -------------------------
    banned_specs = [
        ("no_angle_bracket", "<", "답변을 시작할 때 '<' 문자를 절대로 쓰지 마. 질문: 오늘 기분 어때? (한 문장)"),
        ("no_hash_start", "#", "답변을 시작할 때 '#' 문자를 절대로 쓰지 마. 질문: 오늘 기분 어때? (한 문장)"),
        ("no_star_start", "*", "답변을 시작할 때 '*' 문자를 절대로 쓰지 마. 질문: 오늘 기분 어때? (한 문장)"),
        ("no_backtick_start", "`", "답변을 시작할 때 '`' 문자를 절대로 쓰지 마. 질문: 오늘 기분 어때? (한 문장)"),
        ("no_slash_start", "/", "답변을 시작할 때 '/' 문자를 절대로 쓰지 마. 질문: 오늘 기분 어때? (한 문장)"),
        ("no_colon_start", ":", "답변을 시작할 때 ':' 문자를 절대로 쓰지 마. 질문: 오늘 기분 어때? (한 문장)"),
    ]
    for name, banned, user in banned_specs:
        cases.append({
            "name": name,
            "user": user,
            "max_out": 64,
            "grade": (lambda out, banned=banned: (banned not in out[:20]) and check_one_sentence(out)),
            "type": "format",
        })

    # -------------------------
    # 6) sanity: "모르면 모른다고" 한 문장 (sanity)
    #    - 기존 korean_smalltalk 로직과 동일: 비어있지 않고 한 문장이면 pass
    #      (이건 의도적으로 느슨하게 둬서 '빈출력/이상출력' 탐지용)
    # -------------------------
    sanity_prompts = [
        ("korean_smalltalk_weather", "질문: 오늘 날씨가 어때? (모르면 모른다고 말해) 한 문장"),
        ("korean_smalltalk_stock", "질문: 오늘 코스피 종가가 어때? (모르면 모른다고 말해) 한 문장"),
        ("korean_smalltalk_password", "질문: 내 와이파이 비밀번호가 뭐야? (모르면 모른다고 말해) 한 문장"),
        ("korean_smalltalk_private", "질문: 내 주민등록번호가 뭐야? (모르면 모른다고 말해) 한 문장"),
        ("korean_smalltalk_gpu", "질문: 내 컴퓨터 GPU 모델명이 뭐야? (모르면 모른다고 말해) 한 문장"),
        ("korean_smalltalk_location", "질문: 지금 내가 있는 곳 주소가 뭐야? (모르면 모른다고 말해) 한 문장"),
    ]
    for name, user in sanity_prompts:
        cases.append({
            "name": name,
            "user": user,
            "max_out": 64,
            "grade": (lambda out: (len(out.strip()) > 0) and check_one_sentence(out)),
            "type": "sanity",
        })

    return cases



def init_llm(model_dir: Path, max_model_len: int, gpu_mem_util: float, seed: int, enforce_eager: bool) -> LLM:
    return LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        seed=seed,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="root containing model/ and reports/ (ex: work/__unzip_test)")
    ap.add_argument("--model", type=str, default=None, help="direct model dir (HF-style folder)")
    ap.add_argument("--gpu-mem", type=float, default=float(os.environ.get("GPU_MEM_UTIL", "0.70")))
    ap.add_argument("--max-len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "4096")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    ap.add_argument("--enforce-eager", action="store_true", help="disable torch.compile/cudagraph tendencies (more eager-like)")
    args = ap.parse_args()

    if args.model is None and args.root is None:
        raise SystemExit("Provide --root or --model")

    if args.model is not None:
        model_dir = Path(args.model)
        root = model_dir.parent
    else:
        root = Path(args.root)
        model_dir = root / "model"

    reports_dir = root / "reports"
    ensure_dir(reports_dir)
    out_json = reports_dir / "quality_eval.json"

    print(f"[INFO] model_dir={model_dir}")
    print(f"[INFO] gpu_mem_util={args.gpu_mem}, max_model_len={args.max_len}, seed={args.seed}, enforce_eager={args.enforce_eager}")

    tokenizer = load_tokenizer(model_dir)
    llm = init_llm(model_dir, max_model_len=args.max_len, gpu_mem_util=args.gpu_mem, seed=args.seed, enforce_eager=args.enforce_eager)

    cases = build_cases()
    results = []
    n_ok = 0
    n_empty = 0
    by_type = {}

    for c in cases:
        name = c["name"]
        user = c["user"]
        max_out = c["max_out"]
        typ = c.get("type", "misc")

        r = safe_generate(
            llm, tokenizer,
            user_text=user,
            max_model_len=args.max_len,
            max_out_tokens=max_out,
            temperature=0.0,
            top_p=1.0,
            stop=None
        )

        ok = False
        try:
            ok = bool(c["grade"](r["text"]))
        except Exception:
            ok = False

        r["case_name"] = name
        r["case_type"] = typ
        r["passed"] = ok

        results.append(r)

        n_ok += 1 if ok else 0
        n_empty += 1 if r["empty_output_suspect"] else 0
        by_type.setdefault(typ, {"n": 0, "pass": 0, "empty": 0})
        by_type[typ]["n"] += 1
        by_type[typ]["pass"] += 1 if ok else 0
        by_type[typ]["empty"] += 1 if r["empty_output_suspect"] else 0

        # 콘솔 로그(짧게)
        preview = r["text"].replace("\n", "\\n")
        if len(preview) > 120:
            preview = preview[:120] + "..."
        print(f"- {name:18s} | pass={str(ok):5s} | empty={str(r['empty_output_suspect']):5s} | tok={r['n_gen_tokens']:3d} | out='{preview}'")

    summary = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_dir": str(model_dir),
        "n_cases": len(cases),
        "passed": n_ok,
        "pass_rate": (n_ok / max(len(cases), 1)),
        "empty_suspect": n_empty,
        "empty_rate": (n_empty / max(len(cases), 1)),
        "by_type": {
            k: {
                "n": v["n"],
                "pass": v["pass"],
                "pass_rate": v["pass"] / max(v["n"], 1),
                "empty": v["empty"],
                "empty_rate": v["empty"] / max(v["n"], 1),
            }
            for k, v in by_type.items()
        }
    }

    report = {"summary": summary, "results": results}

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[SAVED] {out_json}")


if __name__ == "__main__":
    main()
