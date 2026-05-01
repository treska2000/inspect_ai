# AI Safety Evaluations with Inspect AI

Hands-on coursework for the **AI Safety Evaluations** course, built on the [Inspect AI](https://inspect.ai-safety-institute.org.uk/) framework from the UK AI Safety Institute.

Four progressive tutorials that go from "run your first eval" to evaluating multi-step ReAct agents on competition math.

---

## Tutorials

| Week | Notebook | Topic |
|------|----------|-------|
| 1 | `inspect_ai_tutorial_week_1.ipynb` | Basics: datasets, solvers, scorers, position bias |
| 2 | `inspect_ai_tutorial_week_2.ipynb` | Statistical evaluation: confidence intervals, model comparison, power analysis |
| 3 | `inspect_ai_tutorial_week_3.ipynb` | Custom evaluations: LLM-as-judge, toxicity classification |
| 4 | `inspect_ai_tutorial_week_4.ipynb` | Agent evaluation: ReAct agents, tool use, MATH-500 benchmark |

### Week 1 — Basics

- Connect Inspect AI to local (Ollama) and cloud models
- Understand the `Task → dataset → solver → scorer` pipeline
- Single-choice and multiple-choice benchmarks
- Analyzing **position bias** in MCQ tasks

### Week 2 — Statistical Rigor

- Load and filter MMLU from HuggingFace
- Confidence intervals (CLT, Wilson) for accuracy
- Paired t-test for model comparison
- Power analysis and minimum detectable effect (MDE)
- Chain-of-thought vs. direct answering: does reasoning help?

### Week 3 — LLM-as-Judge

- Build a classifier–judge pipeline from scratch on the **Jigsaw Toxic Comments** dataset
- Blind vs. non-blind judge templates
- Compare model types (base / instruction-tuned) in classifier and judge roles
- Prompt engineering to reduce failure rates and tune FP/FN balance
- Domain-specific scoring with custom penalty weights

### Week 4 — Agent Evaluation

- Define custom tools (`add`, `subtract`, `multiply`, `divide`, `modular_arithmetic`, `sympy_solve`)
- Compare three solver architectures: plain `generate()`, naive tool loop, ReAct
- Iterate on a dev set without touching the test set (METR elicitation protocol)
- Evaluate a ReAct agent on **MATH-500** with model-graded scoring
- Break down results by subject and difficulty level with Wilson confidence intervals

---

## Stack

- **[Inspect AI](https://inspect.ai-safety-institute.org.uk/)** — evaluation framework
- **[Ollama](https://ollama.ai/)** — local model inference (Llama 2, Qwen 2.5, Qwen 3)
- **HuggingFace Datasets** — MMLU, Jigsaw Toxic Comments, MATH-500
- **SymPy** — symbolic algebra tool for the agent
- Python 3.10+, Jupyter

---

## Setup

```bash
# Install dependencies
pip install inspect-ai openai sympy datasets scipy

# Start Ollama and pull a model
ollama serve
ollama pull llama2          # or qwen2.5:3b-instruct, qwen3:8b

# Launch the log viewer (run from this directory)
inspect view
```

Set API keys for cloud providers in a `.env` file if you prefer not to use Ollama:

```
OPENAI_API_KEY=...
PERPLEXITY_API_KEY=...
SAMBANOVA_API_KEY=...
```

---

## Key Results

| Experiment | Finding |
|---|---|
| Position bias (Week 1) | `llama2` accuracy drops from 46.7% → 33.3% when the correct answer is always at position A |
| Model comparison (Week 2) | `qwen2` vs `llama2` on MMLU-math: gap −0.18, p < 0.001, CI [−0.23, −0.08] |
| Toxicity classifier (Week 3) | "Explicit permission + forced format" prompt reduces classifier failure rate to 0% |
| ReAct agent on MATH-500 (Week 4) | `qwen3:8b` + arithmetic/SymPy tools: **93%** accuracy (95% CI [86.3%, 96.6%]) vs 25% without tools |