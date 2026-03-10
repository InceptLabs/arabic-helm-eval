# EduBench Database Schema

**Database:** edubench
**Host:** incept-rds.c7esqey6q6bf.us-west-2.rds.amazonaws.com
**Port:** 5432
**Total Tables:** 20

---

## Evaluation Pipeline Tables

These tables form the core evaluation workflow:
`eval_models` + `eval_datasets` → `eval_runs` → `eval_samples` ← `eval_job_requests`

### eval_models

Registered models for evaluation. 38 models total — mostly Fireworks-hosted (deepseek-r1, kimi-k2, qwen3, fine-tuned variants) plus OpenAI and Anthropic models.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | integer | NO | Primary key |
| name | text | NO | Model identifier (e.g. `deepseek-r1-0528`, `accounts/fireworks/models/qwen3-235b-a22b`) |
| provider | text | NO | Provider name (`fireworks`, `openai`, `anthropic`) |
| created_at | timestamp with time zone | NO | When the model was registered |

### eval_datasets

Registered evaluation datasets. 17 entries across multiple Arabic benchmarks.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | integer | NO | Primary key |
| name | text | NO | HuggingFace dataset identifier (e.g. `OALL/Arabic_MMLU`, `asas-ai/AraTrust`) |
| split | text | NO | Dataset split (`test`, `train`, `multilingual`) |
| created_at | timestamp with time zone | NO | When the dataset was registered |

**Datasets available:**
- `silma-ai/arabic-broad-benchmark` (test)
- `arbml/CIDAR` (test, train)
- `OALL/Arabic_MMLU` (test)
- `MBZUAI/ArabicMMLU` (test)
- `MBZUAI/human_translated_arabic_mmlu` (test, train)
- `OALL/AlGhafa-Arabic-LLM-Benchmark-Native` (test)
- `MBZUAI/MadinahQA` (test)
- `asas-ai/AraTrust` (test)
- `OALL/ALRAGE` (test, train)
- `mhardalov/exams` (test, multilingual)
- `OALL/Arabic_EXAMS` (test)

### eval_runs

Evaluation run records. 1,319 total runs (1,118 completed, 141 failed, 51 aborted, 9 partial).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | integer | NO | Primary key |
| model_id | integer | YES | FK to eval_models (used in older runs) |
| dataset_id | integer | YES | FK to eval_datasets (used in older runs) |
| judge_model | text | NO | Judge model used for evaluation |
| config_json | jsonb | YES | Legacy config field |
| started_at | timestamp with time zone | NO | Run start time |
| finished_at | timestamp with time zone | YES | Run end time |
| status | text | NO | `completed`, `failed`, `aborted`, `partial` |
| error_message | text | YES | Error details if failed |
| candidate_model | text | YES | Model being evaluated (older schema) |
| model_name | text | YES | Model name (newer schema, used directly) |
| task_name | text | YES | Task/benchmark name (newer schema: `aratrust`, `ALRAGE`, `alghafa`, `arabic_mmlu`) |
| config | jsonb | YES | Run configuration |
| completed_at | timestamp without time zone | YES | Completion timestamp |
| total_samples | integer | YES | Number of samples evaluated |
| metrics | jsonb | YES | Aggregated metrics for the run |
| is_thinking_model | boolean | YES | Whether thinking/reasoning mode was used |
| thinking_stats | jsonb | YES | Thinking mode statistics |

**Completed runs by task:**
| Task | Runs | Total Samples | Avg Samples/Run |
|------|------|---------------|-----------------|
| aratrust | 54 | 27,613 | 521 |
| ALRAGE | 13 | 23,023 | 1,771 |
| alghafa | 6 | 96,120 | 16,020 |
| arabic_mmlu | 4 | 18,253 | 4,563 |

**Judge models used:** gpt-4o, gpt-4.1-mini, claude-sonnet-4, Qwen2.5-72B-Instruct (self-hosted variants), `none`/`no_llm` (auto-grading), `thinking`/`no_thinking` (mode flags), loglikelihood.

**Note:** Schema evolved over time — older runs use `model_id`/`dataset_id` FKs with `candidate_model`, while newer runs embed `model_name`/`task_name` directly.

### eval_samples

Per-sample evaluation results. **1,413,337 rows** — the largest table.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | integer | NO | Primary key |
| run_id | integer | YES | FK to eval_runs |
| sample_index | integer | NO | Index within the run |
| source | text | YES | Data source identifier |
| category | text | YES | Question category |
| subcategory | text | YES | Question subcategory |
| format | text | YES | Question format (see below) |
| instruction | text | YES | Prompt instruction sent to model |
| reference | text | YES | Gold/expected answer |
| prediction | text | YES | Model's extracted answer |
| judge_score | numeric | YES | Judge-assigned score |
| judge_reason | text | YES | Judge's reasoning |
| judge_raw | jsonb | YES | Raw judge response |
| meta | jsonb | YES | Additional metadata |
| created_at | timestamp with time zone | NO | Creation timestamp |
| raw_prompt | text | YES | Full prompt sent to model |
| raw_response | text | YES | Full model response |
| thinking_content | text | YES | Model's reasoning/thinking content (for thinking models) |
| thinking_tokens | integer | YES | Token count for thinking content |
| answer_tokens | integer | YES | Token count for the answer |
| latency_ms | numeric | YES | Response latency in milliseconds |
| extraction_method | text | YES | How the answer was extracted from model output |
| extraction_confidence | numeric | YES | Confidence score of extraction |

**Format distribution:**
| Format | Count |
|--------|-------|
| MCQ_4 | 1,077,582 |
| MCQ_3 | 103,790 |
| MCQ_2 | 103,055 |
| generation | 67,405 |
| multiple_choice | 45,871 |
| MCQ_5 | 5,779 |
| Freeform | 990 |
| MCQ | 689 |
| Classification | 416 |

**Extraction methods (17):** `direct`, `direct_after_think`, `arabic_pattern`, `english_pattern`, `arabic_letter_tail`, `english_letter_tail`, `arabic_header_pattern`, `bold_emphasized`, `full_choice_text`, `answer_word_to_choice`, `last_line_letter`, `separated`, `inside_think_fallback`, `failed`, and model-specific hash IDs.

### eval_job_requests

Async job queue for scheduling evaluations. 138 total jobs.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | bigint | NO | Primary key |
| dataset_name | text | NO | Dataset to evaluate on |
| dataset_configs | ARRAY | YES | Specific dataset configurations |
| split | text | NO | Dataset split |
| model_name | text | NO | Model to evaluate |
| judge_model | text | NO | Judge model to use |
| max_workers | integer | NO | Parallel workers |
| max_tokens | integer | NO | Max tokens per request |
| status | text | NO | `pending`, `running`, `completed`, `failed` |
| created_at | timestamp with time zone | NO | Job creation time |
| started_at | timestamp with time zone | YES | When processing started |
| finished_at | timestamp with time zone | YES | When processing finished |
| last_error | text | YES | Most recent error |
| runs | jsonb | YES | Associated run details |
| config_json | jsonb | YES | Job configuration |
| error_message | text | YES | Error details |

**Current job status:**
| Dataset | Completed | Failed | Pending | Running |
|---------|-----------|--------|---------|---------|
| AraTrust | 8 | 8 | 10 | - |
| ArabicMMLU | 10 | 11 | 17 | 9 |
| AlGhafa | 15 | 28 | - | 1 |
| ALRAGE | 11 | 1 | - | 3 |
| Arabic_MMLU | 4 | 1 | - | 1 |

---

## Benchmark-Specific Tables

### alghafa_eval_samples

Evaluation samples specifically for the AlGhafa Arabic LLM benchmark.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | bigint | NO | Primary key |
| run_id | integer | YES | FK to eval_runs |
| model_idf | text | YES | Model identifier |
| category | text | YES | Benchmark category |
| category_idx | integer | YES | Category index |
| sub_category | text | YES | Sub-category |
| expected_answer | integer | YES | Expected answer index |
| question | text | YES | Question text |
| option_texts | ARRAY | YES | Answer options |
| response_raw | text | YES | Raw model response |
| reasoning | text | YES | Model's reasoning |
| predicted_answer | integer | YES | Model's predicted answer index |
| arabic_percentage | double precision | YES | Percentage of Arabic in response |
| created_at | timestamp without time zone | YES | Creation timestamp |

### alghafa_audit_results

Audit/quality review of AlGhafa evaluation results.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | bigint | NO | Primary key |
| primary_category | text | NO | Primary benchmark category |
| secondary_category | text | YES | Secondary category |
| model_reasoning_assessment | text | NO | Assessment of model's reasoning quality |
| benchmark_expected_answer_is_correct | text | NO | Whether the benchmark's expected answer is correct |
| predicted_option_is_correct | text | NO | Whether the model's prediction is correct |
| failure_attribution | text | NO | What caused the failure (model, benchmark, ambiguous) |
| created_at | timestamp with time zone | YES | Creation timestamp |
| eval_model_name | text | YES | Model that was evaluated |

---

## RAG Evaluation Tables

### alrage_analysis

Analysis of ALRAGE (Arabic Language RAG Evaluation) test dataset.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| index_id | integer | NO | Primary key |
| question | text | NO | Question text |
| gold_answer | text | YES | Gold/expected answer |
| candidates | text | YES | Candidate answers |
| reasoning_types | ARRAY | YES | Types of reasoning required |
| context_structure | text | YES | Structure of provided context |
| answer_position | text | YES | Where the answer appears in context |
| is_answerable | boolean | YES | Whether the question is answerable |
| has_noise | boolean | YES | Whether context contains noise |
| noise_level | text | YES | Level of noise in context |
| analysis_trace | text | YES | Full analysis trace |
| created_at | timestamp with time zone | YES | Creation timestamp |

### alrage_train_dataset_analysis

Same schema as `alrage_analysis`, but for the training split of the ALRAGE dataset.

---

## Question Generation Tables

### question_recipes

Templates/recipes for generating educational questions.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| recipe_id | integer | NO | Primary key |
| source_sheet | text | YES | Source spreadsheet |
| grade_level | text | YES | Target grade level |
| subject | text | YES | Subject area |
| domain | text | YES | Knowledge domain |
| cluster | text | YES | Topic cluster |
| standard_id_l1 | text | YES | Level 1 standard ID |
| standard_desc_l1 | text | YES | Level 1 standard description |
| standard_id_l2 | text | YES | Level 2 standard ID |
| standard_desc_l2 | text | YES | Level 2 standard description |
| substandard_id | text | YES | Sub-standard ID |
| lesson_title | text | YES | Lesson title |
| question_type | ARRAY | YES | Types of questions to generate |
| tasks | text | YES | Task descriptions |
| difficulty | text | YES | Target difficulty level |
| constraints | text | YES | Generation constraints |
| direct_instruction | text | YES | Direct instruction text |
| step_by_step_explanation | text | YES | Step-by-step explanation |
| misconception_1 | text | YES | Common misconception 1 |
| misconception_2 | text | YES | Common misconception 2 |
| misconception_3 | text | YES | Common misconception 3 |
| misconception_4 | text | YES | Common misconception 4 |
| created_at | timestamp with time zone | YES | Creation timestamp |
| misconceptions | text | YES | Combined misconceptions |
| refined_difficulty | jsonb | YES | Refined difficulty assessment |

### generated_questions

AI-generated questions from recipes.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | integer | NO | Primary key |
| recipe_id | integer | YES | FK to question_recipes |
| model | text | YES | Generation model |
| inference_params | jsonb | YES | Model parameters used |
| prompt_text | text | YES | Prompt sent to model |
| model_raw_response | text | YES | Raw model output |
| model_parsed_response | jsonb | YES | Parsed/structured output |
| created_at | timestamp without time zone | YES | Creation timestamp |
| question_type | text | YES | Type of question generated |
| experiment_tracker | text | YES | Experiment tracking ID |
| embedding | USER-DEFINED | YES | Vector embedding (pgvector) |
| embedding_model | text | YES | Model used for embedding |
| image_uri | text | YES | S3 URI for associated image |
| image_url | text | YES | Public URL for image |
| image_prompt | text | YES | Prompt used to generate image |
| updated_at | timestamp without time zone | YES | Last update timestamp |

### ai_evaluation_results

AI evaluator scores for generated questions.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | integer | NO | Primary key |
| question_id | integer | YES | FK to generated_questions |
| evaluator_version | text | YES | Evaluator model/version |
| evaluator_raw_response | text | YES | Raw evaluator output |
| evaluator_parsed_response | jsonb | YES | Parsed evaluation |
| evaluator_score | double precision | YES | Numeric score |
| evaluated_at | timestamp with time zone | YES | Evaluation timestamp |
| evaluator_tracker | text | YES | Experiment tracking ID |
| evaluator_input | jsonb | YES | Input sent to evaluator |

### question_image_requirements

Flags indicating which questions require accompanying images.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| q_id | bigint | NO | FK to generated_questions |
| need_image | boolean | NO | Whether an image is needed |

### image_generations_test

Test results for image generation pipeline.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| question_id | bigint | YES | FK to generated_questions |
| requires_image | boolean | YES | Whether image is required |
| image_gen_prompt_builder_prompt | text | YES | Prompt used to build the image prompt |
| image_gen_prompt | text | YES | Final image generation prompt |
| image_gen_model | text | YES | Image generation model |
| image_s3_uri | text | YES | S3 URI of generated image |
| created_at | timestamp without time zone | YES | Creation timestamp |
| nanobanana_image_s3_uri | text | YES | Alternative image S3 URI |
| is_svg | boolean | YES | Whether output is SVG |
| evaluator_tracker | text | YES | Experiment tracking ID |
| svg_code | text | YES | SVG source code |

---

## Question Bank Tables

### mcq_qas

Multiple-choice question bank.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | integer | NO | Primary key |
| source_dataset | varchar | YES | Source dataset |
| question_type | varchar | YES | Type of question |
| listed_grade | integer | YES | Original grade level |
| listed_grade_level | varchar | YES | Grade level label |
| listed_difficulty | double precision | YES | Original difficulty |
| estimated_grade | integer | YES | AI-estimated grade |
| estimated_grade_level | varchar | YES | AI-estimated grade label |
| estimated_difficulty | double precision | YES | AI-estimated difficulty |
| question | varchar | NO | Question text |
| option_a | varchar | NO | Option A |
| option_b | varchar | NO | Option B |
| option_c | varchar | YES | Option C |
| option_d | varchar | YES | Option D |
| correct_answer | varchar | NO | Correct answer |
| explanation | varchar | YES | Answer explanation |
| created_at | timestamp without time zone | NO | Creation timestamp |
| option_e | varchar | YES | Option E |
| option_f | varchar | YES | Option F |
| option_g | varchar | YES | Option G |
| estimated_subject | varchar | YES | AI-estimated subject |
| context | varchar | YES | Additional context |

### open_ended_qas

Open-ended question bank.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | integer | NO | Primary key |
| source_dataset | varchar | YES | Source dataset |
| question_type | varchar | YES | Type of question |
| listed_grade | integer | YES | Original grade level |
| listed_grade_level | varchar | YES | Grade level label |
| listed_difficulty | double precision | YES | Original difficulty |
| estimated_grade | integer | YES | AI-estimated grade |
| estimated_grade_level | varchar | YES | AI-estimated grade label |
| estimated_difficulty | varchar | YES | AI-estimated difficulty |
| estimated_subject | varchar | YES | AI-estimated subject |
| listed_subject | varchar | YES | Original subject |
| question | varchar | NO | Question text |
| correct_answer | varchar | YES | Expected answer |
| explanation | varchar | YES | Answer explanation |
| context | varchar | YES | Additional context |
| created_at | timestamp without time zone | NO | Creation timestamp |

---

## Content Tables

### book_chapters

Arabic textbook chapters used for question generation.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | integer | NO | Primary key |
| chapter_id | text | YES | Chapter identifier |
| book_id | text | NO | Book identifier |
| book_title | text | YES | Book title |
| chapter_order | integer | YES | Chapter sequence number |
| chapter_title | text | YES | Chapter title |
| subject | text | YES | Subject area |
| estimated_grade | integer | YES | Estimated grade level |
| estimated_grade_level | text | YES | Grade level label |
| estimated_difficulty | text | YES | Difficulty label |
| word_count | integer | YES | Word count |
| char_count | integer | YES | Character count |
| text | text | NO | Full chapter text |
| meta | jsonb | YES | Additional metadata |
| created_at | timestamp without time zone | YES | Creation timestamp |

---

## Infrastructure Tables

### api_tokens

API authentication tokens.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | integer | NO | Primary key |
| token | varchar | NO | API token value |
| client_name | varchar | YES | Client identifier |
| is_active | boolean | YES | Whether token is active |
| created_at | timestamp without time zone | YES | Creation timestamp |
| app_name | text | YES | Application name |

### incept_api_logs

API request/response logging.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | integer | NO | Primary key |
| request_timestamp | timestamp without time zone | YES | When request was received |
| source_ip | varchar | YES | Client IP address |
| user_agent | text | YES | Client user agent |
| method | varchar | YES | HTTP method |
| url | text | YES | Request URL |
| request_body | text | YES | Request payload |
| response_body | text | YES | Response payload |
| status_code | integer | YES | HTTP status code |
| process_time_seconds | double precision | YES | Processing time |
| client_name | varchar | YES | Authenticated client name |
| prompt | text | YES | Extracted prompt text |

### leaderboard_results

Aggregated leaderboard scores.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| model | text | YES | Model name |
| subject | text | YES | Subject area |
| question_type | text | YES | Question type |
| difficulty | text | YES | Difficulty level |
| questions_above_threshold | bigint | YES | Questions scoring above threshold |
| total_questions | bigint | YES | Total questions evaluated |
| difficulty_sort_key | integer | YES | Sort order for difficulty |
