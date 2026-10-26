# 오늘 Full 실험 진입을 위한 실행 가이드라인 v2

작성 2026-09-20. 사용자 목적: 기존 방법을 유지하면서 리비전을 해결하고, 오늘 Full 실험 실행까지 진행한다. 구현 담당은 현재 세션/사용자가 전환한 Sol이다. 이 문서는 실행 판단 지침이며 실행 완료 증거가 아니다.

## 1. 판단과 적용 순서

오늘의 목표를 `실행 가능한 소스 확정 → 실제 한 사례 연결 검사 → 사전 고정한 Full/BASE 본 실행 제출`로 둔다. 기존 감사 전부를 다시 하거나 좋은 출력이 나올 때까지 모델을 조정하는 단계를 넣지 않는다. 소스/입력/후보 회계가 검증되면 이미 받은 사용자 진행 지시에 따라 다음 단계로 간다. 문서 서식이나 외부 AI의 추가 승인은 제출 선행조건이 아니다.

다만 오늘 완료 또는 성공을 보장하지 않는다. 남은 수정량·실제 GPU 메모리·Slurm 대기·평가 비용을 아직 측정하지 않았다. 같은 조건의 실행 실패를 반복해서 하루를 소모하지 말고, 실패 지점과 재현 반례를 특정해 관련 수정만 한다.

이 문서는 9/20 `one_case_original_method_integration_v1.json`과 직전 가이드의 일괄 중단 규칙을 대체한다. 원 v1과 과거 프로토콜은 보존한다. **v1 JSON의 hash가 고정됐다는 사실은 실행 정책 검증 완료를 뜻하지 않는다.** 실행자는 v2 근거에 따른 새 executable contract를 생성한다.

원 방법 판정은 `원고의 제안 정의 + 보관 구현 + 해당 실행 계보 + 확인된 오류 반례`를 함께 쓴다. reference_main은 비교 자료이며 모든 과거 실행의 정확한 바이트로 인증된 것은 아니다. 별도 snapshot을 만드는 것만으로 방법 동일성이 성립하지 않는다.

## 2. 이미 끝난 작업은 재사용한다

- 보존 후보3,132건, controls216묶음, Error→Fail587건(실행 오류 귀속543+실행 전 문법44), 잔여Error297의 연결 검증을 재사용한다. 기존 후보 전면 재평가를 새 Full 진입 조건으로 삼지 않는다.
- v4 fresh 학습은 완료됐다. `planner_archived_rng_primary_v4/bd4dc70ca4206e17/best_model_state_dict.pt`, SHA `b02f95aa1fcb24b19f5a5e038e47976125ce7979a686821810465c3f4ee2f730`. 새 학습 필요성을 입증하는 결함이 없으면 재학습하지 않는다.
- 기존 지표 교정의 의도는 타당하다: prefix 안에 Pass가 있으면 success, Pass 없이 Error/Timeout이면 unknown. @1/@5/@10의 기존 Pass 분자는 전부 불변이었다. 실제 실행 스키마와 연결하는 남은 검사는 §7에 둔다.
- `16 passed, 14 skipped`에서 14개는 해당 로컬 환경의 torch 부재로 skip됐으며 완료 검증으로 세지 않는다. 직전 보고의 단순 “optional” 표현은 정정한다. 앞으로 변경한 경로가 그 검사에 의존하면 필요한 환경에서 해당 검사만 수행한다.
- 현재 증거와 후속 결과를 `tmp/2026_09_20/pro_design_verification_v2/` 및 WORK_LEDGER에 연결한다. 변경이 없는 범위의 같은 감사는 반복하지 않는다.

## 3. 변경별 실행 판단

| 대상 | 이번 실행의 결정 | 증명/공개할 사항 |
|---|---|---|
| 학습 및 planner | 검증된 v4 best, 보관6-op/type map, 기존 학습·추론 설정 유지 | 학습 L32/추론 L16, noise·guidance 불일치는 실제 설정으로 공개. 이를 개선하기 위한 변경은 하지 않음 |
| 모델 역할 | 표 연결 근거의 StarCoder2를 AST/patch에 사용, E5 검색 유지 | Qwen 결과는 별도 진단. 과거 weights exact identity 부재와 현재 사용 weights identity를 구분 |
| AST 추가 품질 검사 | 단어수·응답 line·설명 품질을 관측값으로 분리. all10-valid/valid7 선택을 primary gate로 사용하지 않음 | 질의→실제 AST 노드 연결과 반환 line을 별도 저장. returned line이 다르다는 사실만으로 실제 주소가 바뀌었다고 단정하지 않음 |
| AST 실패 응답의 소비 | 원 extractor/질의순서/consumer의 실제 행동을 고정하고 품질 표시를 별도로 저장 | 실패 문자열을 valid 설명으로 표시하거나 지우지 않음. 원 소비 경로에 포함된다면 그 사실을 공개하고 해당 실행을 정상 AST 추론의 증거로 쓰지 않음 |
| sampler support | 참조1024 support/smoothing/난수 소비를 isolated 경로에 유지 | 추가 source-node mask·투영·재표집 제외. KEEP의 사용되지 않는 node, 버려지는 계획, 실제 전달 편집을 구별 |
| 후보 syntax | 참조 ast.parse 기반 생성 수용 정책 유지, compile 결과는 별도 진단 | generation의 compile 필터로 후보 제거/추가 생성하지 않음. evaluator의 compile 판정은 기존 평가 계약 유지 |
| 계획 준수 필터/점수 | 실제 원 filter/bonus/정렬/추가생성 규칙을 기본으로 고정 | 중복 snippet의 잘못된 줄 결박 등 근거 있는 주소 교정은 실제 반례·최소 delta·영향을 갖춘 경우에만 별도 적용. line→subtree 전체로 확대 금지 |
| 검색 | 원 코퍼스·BM25/dense 결합·순서 유지, 실행별 새 cache | 자기 ID 타입 정규화/캐시 identity 등 검증된 연결 오류 교정 가능. 새 pair dedup·backfill은 기본 제외 |
| target 정답 경계 | 기존 answer-free input 교정 유지 | target fixed/commit/fix diff/evaluation 필드가 모델 입력에 유입되지 않음. 참조 prompt 생성기를 무조건 복원하면 commit info가 다시 유입될 수 있음 |
| tokenizer와 출력 파싱 | 원 template/4096 제한/기존 절단 정책 및 actual decoder 경로를 확인해 고정 | 전체 decode와 completion-only 차이, padding 길이·prompt echo를 반례로 검증. 오류 증거 없이 추출기를 새로 선택하지 않음 |
| 평가와 표 | 삼상태·raw/final·분모 공개 교정 유지 | 후보 생성/학습이 같아도 평가/보고가 수정됐다는 사실과 영향은 공개 |

학습 원자료에는 Delete263562/Replace59183/InsertBefore592/InsertAfter97이 있고 미지원 op는 관측0이었다. raw 라벨에 GuardAdd가 없다는 사실만으로 GuardAdd head가 “전혀 학습되지 않았다”고 쓰지 않는다. softmax 음성 gradient와 corruption 입력 노출이 가능하며, 검증된 표현은 “보존 라벨의 양성 GuardAdd 예제가 관측되지 않음”이다.

## 4. 중단 조건을 정확히 제한한다

### 전체 실행 중단이 필요한 오류

- 실제 로드한 코드/가중치/metadata/image가 동결 manifest와 다름.
- 실행자가 원 설정 대신 다른 모델·프롬프트·선택 정책을 사용했거나 조건 간 후보·결과가 섞임.
- target 정답/평가 정보가 생성 경계로 넘어감, 후보 hash와 실제 평가 패치 불일치, 복원 무결성 실패.
- 전 작업에 영향을 주는 원인 불명의 런타임/계측 오류. 수정 시 기존 부분 결과를 보존하고 새 버전으로 분리.

### 사례별로 기록할 상태

- 원래 방법의 AST 파싱 실패, 설명 품질 부족, 빈 계획/후보, OOM/시간 초과, 주소를 해석할 수 없는 실제 소비 편집 등은 발생 단계와 원시 출력으로 기록한다. 해당 버그를 분모에서 삭제하지 않는다.
- 실제 소비 편집의 노드 주소가 틀리면 그 사례의 유효한 Full 실행을 인증할 수 없다. 원래 consumer의 처리 규칙을 확인하고, fail-stop이 추가되는 경우 그 편차 및 실행 coverage를 공개한다. 사용되지 않는 KEEP node 하나 때문에 새로 중단시키지 않는다.
- 검색에서 `target BEFORE == 다른 AFTER`만으로 target fixed 누출이라고 결론내리지 않는다. 자기 ID, 실제 정답 provenance, 허용 코퍼스 정책 위반 여부를 구별한다. 문자열 일치는 관계 관측이며 기존 adapter의 `target_source_match` 일괄 판정은 그대로 사용하지 않는다.
- 4096 초과는 원 truncation 동작과 최종 토큰을 기록한다. 원래 허용된 잘림을 blanket fail로 바꾸면 새 수용 정책이 된다. task/source identity를 확인할 수 없는 경우는 input 제한 사례로 별도 기록하고 길이 확대·재요약하지 않는다.
- 부분 실패가 있어도 유효하게 정의된 다른 작업은 계속할 수 있다. 다만 한 사례 통합 검사에서 어떠한 Full 하류 연결도 확인하지 못했다면 전체 제출을 성공한 통합 검사로 표시하지 않는다.

### 단계 통과 조건에 넣지 않을 것

all10-valid, 정확히12 plans, 정확히10 candidates, Pass>0, BASE보다 좋은 점수, 유의성, 모든 런타임 오류0건. 실패가 결과로 보존됐는지와 코드가 동결된 방법을 실행했는지가 기준이다.

## 5. 오늘 생성할 실험 계약

주 비교는 **Full 대 BASE, seed42/43/44, V1 strict43**이다. 이는 제안 시스템 전체 비교다. 계획만의 인과 효과, 검색만의 효과, SOTA 우월성, 두 모델 일반화를 이 한 비교로 주장하지 않는다. 원고에 그 별도 주장을 유지하려면 대응 비교를 결과 확인 전에 별도 설계해야 한다.

- V1 정본: `/cephfs/home/s2022810061/tmp/2026_09_17/kcse_control_eligibility_v1.json`, SHA `6fb8d7bc38e8a1be4fe757c6c095af20d80e3ce74c1df6b240491d1f61713dd1`. manifest의43개 ID와 사례별 control 근거를 재사용·결박한다. 유리한 결과를 보며36개로 바꾸지 않는다. Vraw41 및 원51 coverage를 함께 보존한다.
- 모델·planner·입력자료·자원·생성/평가 source·이미지·조건·seed·예산·timeout·재시도·분석 단위를 새 실행 manifest에 기록한다. manifest의 hash를 생성 전에 저장한다.
- 총 `43 × 2 × 3 = 258`개 bug-arm-seed 시도. 각 seed 결과를 따로 보고, 기존3-seed 병합을 추가 보고하면 정렬/tie-break를 생성 전에 고정한다. 3개의 seed를 R1/R2/R3 또는 독립 planner 재학습으로 부르지 않는다.
- bug/seed당 P1, target 최대10, greedy1회+sampling 최대100, batch8, 기존 grid+jitter, 최대640 생성 토큰. T=.8/p=.9는 기본값이며 모든 sampling call이 같은 값이라는 뜻은 아니다. compliant9+fallback1은 참조 정책대로9; fallback 보충으로10을 만들지 않는다.
- AST/planner/patch의 실제 RNG 설정 위치를 고정한다. 계측이 난수를 추가 소비하지 않아야 한다. Full과 BASE의 같은 seed가 동일 token sampling stream을 보장한다고 주장하지 않는다.
- BASE는 동일 bug/task/test/generator/공통 비계획 selector를 사용한다. AST/prior/retrieval/planner/계획 문장/raw edits/plan filter/bonus가 최종 입력과 선택 경로에 없는지 검사한다. Full은 원래 계획 전용 효과를 유지한다. 두 arm에서 모든 selector 항이 동일하다고 쓰지 않는다.
- Full과 BASE는 독립 생성하므로 raw pool이 같다는 요구를 하지 않는다. 기존 noRanking/noRetrieval을 BASE로 바꾸어 이름 붙이지 않는다.
- 시간/메모리/재시도 한계는 통합검사 자원 관측 후 모든258시도의 생성 전 고정한다. 정상 생성 실패나 낮은 점수를 이유로 재시도하지 않는다. 명확한 일시적 인프라 실패만 같은 입력/seed/소스/한계로 제한 재시도하며 최초 시도를 남긴다.

## 6. 실행 순서와 완료물

### A. 별도 실행 소스 만들기

이미 frozen인 파일/dirty worktree/원고를 덮어쓰지 않는다. `revision_full_20260920_v2` 같은 새 디렉터리에 참조 모듈과 승인된 최소 교정을 복사한다. copied source 경로를 명시적으로 받게 하고 `MAIN=ROOT.parent.parent`처럼 위치에 따라 다른 참조 파일을 읽는 구조를 제거한다.

출력은 실행 소스와 `CHANGE_DECISIONS.json`이다. 각 delta에 원 의도 근거, 이전/이후 함수 hash, 반례, 분포/후보/입력 영향, 원고 공개 문구를 넣는다. 정책의 이유가 적혀 있어도 실행 코드가 구현하지 않았으면 완료가 아니다.

### B. 실행 전 필요한 CPU 차등검사만 수행

실제 참조/새 함수에 같은 입력·RNG·가짜 model outputs를 주어 sampler 호출/분포, AST 순서, 후보 순서/수용/추가 생성, 검색 결과, 최종 prompt를 대조한다. tokenizer의 실제 template/token IDs 대조는 실제 모델 환경에서 한다.

필수 반례:21단어·returned line 불일치와 실제 node identity 구별; KEEP dummy node;36draw 중복1계획; parse 가능 compile 불가; compliant9+fallback1; 같은 길이 다른 corpus cache; 자기 ID2/"2"; target BEFORE 일치와 정답 누출 구별;4096 초과; BASE에 숨은 계획 문장/bonus; empty candidate 대 missing evaluation.

동등성을 주장하는 경로는 결과뿐 아니라 RNG 소비·시도 수·후보 순서가 맞아야 한다. 허용된 오류 교정은 달라지는 반례와 정상 입력 불변을 확인한다. fixture mock 통과를 실제 end-to-end 통과라고 부르지 않는다.

### C. bug2/seed42 Full+BASE 통합검사

같은 frozen source로 실제 Slurm GPU1에서 analyzer→retrieval→planner→최종 tokenizer→candidate까지 연결하고 평가도 실제 CPU Slurm 작업에서 확인한다. 원 AST-only1680/partial-valid7 실험과 달리 원 소비 정책 및 두 arm의 실제 입력·후보 회계가 검사 질문이다. valid7 결과를 재사용해 Full을 구성하지 않는다.

검사 후 필요한 것은 start/end manifest, image ID/inspect, 단계별 raw 출력, 질의-node 관계, 실제 토큰, 후보 수용·제외 사유, 후보 hash 및 원시 평가다. 코드가 계약대로 연결되고 결과·실패 상태가 완전하면 Pass0이어도 통합검사를 닫을 수 있다. 상류 실패만 반복되고 하류를 실행하지 못한 상태는 “하류 검증 완료”가 아니다.

통합검사와 본 실행 계약이 완전히 같으면 해당 두 시도를258개에 재사용한다. 변경이 생겼다면 이전 결과는 진단으로 보존하고 새 계약 결과와 혼합하지 않는다.

### D. Full/BASE 본 실행 시작 및 평가 연결

A–C의 실제 근거가 갖춰지면 새로운 외부 검토를 요청하며 대기하지 말고 사용자 진행 지시대로 본 실행을 제출한다. 사례별 결과가 봉인되는 대로 평가를 연결해 생성/평가 대기 시간을 줄인다. 각 task/arm/seed의 output/checkouts/cache를 분리한다.

성적에 따른 조기종료·seed추가·후보예산 변경을 금지한다. 모든 예정 시도가 결과/실패/미실행 중 무엇인지 끝까지 추적한다. 전체 시작, 일부 완료, 전체 완료를 구분해 job ID와 함께 보고한다.

## 7. 평가·통계·문서에서 유지할 구별

- `Success`라는 실행 status 하나로 실제 후보 평가를 인증하지 않는다. 후보 개수/순서/hash, fixed→buggy→restored-fixed, candidate restore, raw/final flags를 대조한다. evaluator는 기존 승인된 pandas 복구·timeout·attribution 규칙을 유지한다.
- **후보0개로 완료한 생성**, **실행/파일 결측**, **실제로 평가한 Fail**, **미판정**을 구별한다. 최근 prefix helper는 빈 flags를 failure로 취급하므로 호출 전에 이것이 진짜 후보0개인지 검증해야 한다.
- 기존 `value=S/evaluable` 필드는 unknown이 존재하면 확정 성공률이 아니다. 새 집계에서는 전체 예정 분모N, S/F/U, stage별 coverage, S/N 및 가능한(S+U)/N을 명시한다. code의 이름만 보고 그대로 표에 넣지 않는다. N과 manifest ID수가 일치하는지도 검사한다.
- DeepSeek24/36은 보존 자료의 관측 yield다. 30/36은 결측6개가 모두 성공했다는 가정 아래 성공 여부의 가능 상한이다. 실제 측정 yield를30/36으로 수정한 것은 아니다.
- paired unknown 제외 후 검정은 complete-pair 조건부 분석이다. excluded ID와 전체 분모의 불확실성을 별도 보고한다. 여러 seed를 독립 bug로 세지 않고 bug를 묶어 재표집한다.
- 원51 전체에 대한 control coverage43/51, Vraw41 민감도, 기존V0 및 보존36 결과를 구분한다. 과거 성공 수를 새43분모로 나누지 않는다.
- AST 실패를 원 경로가 소비했다면 실제 수행한 시스템 결과로 보고할 수 있으나 정상 설명/추론의 기여가 입증됐다고 쓰지 않는다. 새 비교가 부정적이면 그 결과에 맞게 주장 범위를 조정한다.
- 문서 정정(raw/final, P1/P3,6ops,최대12,모델 역할,PF 해석,수식/표 표시)은 실행과 독립적으로 진행한다. 실행 진입은 제출 준비 완료와 별개다.

## 8. CEHUB 운영과 오늘 보고 기준

Docker/Compose는 실제 SLURM_JOB_ID가 확인된 작업 안에서만 사용한다. GPU는 Slurm `--gres=gpu:1`, 컨테이너 `--device=nvidia.com/gpu=all`. `cehub-docker images/load/save` 규칙을 따르고 CephFS 저장 이미지를 먼저 복원한다. sudo·Docker socket/group 우회는 없다.

동시 사용 worker node는 사용자 다른 세션 작업까지 합쳐 최대5, 기존 평가 task 동시성은4 유지. 생성/평가가 자원을 중복 요청하지 않도록 현재 Slurm 상태를 제출 직전에 확인한다. 노드명/특정 GPU를 강제로 점유하는 우회는 하지 않는다. pilot에서 충분한 메모리와 처리량을 확인한 후 필요한 자원만 요청한다.

보고 상태는 `SOURCE_READY`, `PILOT_RUNNING`, `PILOT_ACCOUNTED`, `FULL_SUBMITTED`, `FULL_RUNNING`, `FULL_COMPLETE`와 실제 근거/다음 행동으로 남긴다. 파일만 만들고 SOURCE_READY 또는 FULL_SUBMITTED라고 보고하지 않는다. 지금 이 문서를 작성한 시점에는 v2 실행 소스·통합검사·본 실행이 아직 완료되지 않았다.
