# One-click Eval Tools (vLLM)

EXAONE-4.0-1.2B 계열 모델을 **HF 폴더 그대로(vLLM 로딩 기준)** 빠르게 점검/비교하기 위한 스크립트 2종입니다.

- `eval_oneclick.py`  
  → **패키징 체크 + 스모크 테스트 + 간단 속도 벤치** (토큰당 시간/초당 토큰)

- `quality_eval_oneclick.py`  
  → **간이 “품질/지시준수” 점검용 케이스**(반복문자, JSON만 출력, 한 문장, 금지문자, 간단 산수, sanity 등) 통과율 측정  
  ⚠️ 대회 Public/Private 점수와 직접적으로 일치하지 않습니다. (회귀/깨짐 탐지용)

---

## 권장 폴더 구조

### (권장) submit.zip 테스트 언집 후

work/__unzip_test/
model/ # HF 표준 모델 폴더 (config/weights/tokenizer 등)
reports/ # 스크립트가 자동 생성
tools/
eval_oneclick.py
quality_eval_oneclick.py
README.md


> `eval_oneclick.py`는 `--root` 지정 시 root 아래에서 **config.json + weights**가 있는 “그럴듯한 폴더”를 자동 탐색합니다.  
> `quality_eval_oneclick.py`는 `--root` 지정 시 **root/model**을 사용합니다.

---

## 요구 환경

- Python 3.11
- `vllm` (대회 서버 기준: 0.14.1)
- `transformers` (대회 서버 기준: 4.57.3)
- CUDA 가능한 GPU

> 로컬과 서버 환경이 다르면 속도 수치는 달라질 수 있습니다. (상대 비교용으로 사용)

---

## Quick Start

### 1) 패키징 체크 + 스모크 + 속도 벤치
```bash
python tools/eval_oneclick.py \
  --root work/__unzip_test \
  --gpu-mem 0.70 \
  --max-len 4096

또는 모델 폴더 직접 지정:

python tools/eval_oneclick.py \
  --model work/__unzip_test/model \
  --gpu-mem 0.70 \
  --max-len 4096
```


###2) 간이 품질/지시준수 체크(통과율)
```bash
python tools/quality_eval_oneclick.py \
  --root work/__unzip_test \
  --gpu-mem 0.70 \
  --max-len 4096
```
## 주요 옵션(공통/유사)
```
--root : 언집한 root 디렉토리 (보통 .../__unzip_test)

--model : HF 모델 폴더 직접 지정

--gpu-mem : vLLM gpu_memory_utilization (기본 0.70, 환경변수 GPU_MEM_UTIL로도 지정 가능)

--max-len : vLLM max_model_len (기본 4096, 환경변수 MAX_MODEL_LEN로도 지정 가능)

--seed : vLLM seed (기본 0, 환경변수 SEED 가능)

--enforce-eager : eager-like 실행(컴파일/그래프 최적화 영향 줄이기)

eval_oneclick.py 전용:

--no-fix-mistral-regex : tokenizer 로딩 시 fix_mistral_regex 시도를 끔
(transformers 버전에 따라 경고/충돌이 있을 때 사용)
```

---

## 출력 파일
eval_oneclick.py

```

저장 위치: (<model_dir>/../reports)/eval_report.json

포함 내용:

package summary: config/weights/tokenizer/chat_template 등 필수 파일 존재 여부

smoke: 3개 스모크 케이스 결과 + empty_output_suspect 플래그

bench: short/mid/long 입력에서 대략적인 tok_per_sec, sec_per_token

콘솔에서 요약이 같이 출력됩니다:

[SMOKE] results : EMPTY? / OK

[BENCH] results : tok/s, sec/tok
```
---

## quality_eval_oneclick.py
```
저장 위치: (<root>/reports)/quality_eval.json

포함 내용:

summary.pass_rate : 전체 케이스 통과율

summary.empty_rate : 빈 출력(또는 EOS로 시작) 의심 비율

summary.by_type : format / accuracy / sanity 타입별 통과율

results[] : 케이스별 출력/토큰 수/통과 여부/프롬프트 truncate 여부 등
```

## 해석 가이드 (실험할 때 “어디가 깨졌는지” 빨리 찾기)
### 1) empty_output_suspect=True가 많이 뜬다
```
채팅 템플릿 문제(apply_chat_template / eos 처리) 또는

양자화/가중치 깨짐(특정 레이어/헤드/embedding 누락 등) 가능성

→ 먼저 eval_oneclick.py의 package summary에서

config_json, has_weights, tokenizer_json/tokenizer.model,

chat_template.jinja 존재 여부 확인
```
### 2) prompt_truncated=True가 자주 뜬다
```
max_model_len이 너무 작거나 입력이 길어서 잘린 것

긴 컨텍스트 성능/정확도 테스트는 --max-len을 키워야 함 (OOM 주의)
```
### 3) 벤치(tok/s)는 빠른데 quality pass_rate가 급락
```
“정답/형식 준수”가 무너진 경우가 많음 (특히 GPTQ ignore 범위/캘리브레이션 데이터 영향)

회귀 확인용으로 quality_eval_oneclick.py를 항상 같이 돌리는 걸 추천
```
## 트러블슈팅
### (A) incorrect regex pattern 경고
```
실행 자체는 되지만 토크나이즈가 미묘하게 틀어질 수 있음.

eval_oneclick.py는 가능한 경우 fix_mistral_regex=True를 시도합니다.

충돌나면 --no-fix-mistral-regex로 끄고 진행하세요.
```
### (B) OOM / CUDA out of memory
```
--max-len 낮추기 (예: 4096 → 2048)

--gpu-mem 낮추기 (예: 0.70 → 0.60)

--enforce-eager 켜보기 (환경에 따라 메모리 사용 패턴이 달라질 수 있음)
```
##실험 루틴 추천(해커톤용)
```
submit.zip → 언집(work/__unzip_test/model)

eval_oneclick.py로 패키징/빈출력/속도 1차 확인

quality_eval_oneclick.py로 형식/간이정확도/지시준수 회귀 확인

결과 JSON(reports/*.json)을 모아 CSV로 정리하면 모델 스윕이 쉬워짐
```
