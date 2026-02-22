# Multi-Tool Personal Research Agent (LangChain)

A LangChain + Streamlit project that researches a topic using multiple tools:
- Web search (DuckDuckGo)
- arXiv paper lookup
- Wikipedia background lookup
- Project-based local note memory (JSON)
- Source reliability tags in briefs (`[High]`, `[Medium]`, `[Low]`)
- Export research briefs as Markdown and PDF
- Model fallback handling when the configured model is unavailable

## 1) Setup

```bash
cd multi_tool_research_agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Environment

```bash
copy .env.example .env
```

Set your key in `.env`:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

## 3) Run

```bash
streamlit run app.py
```

## What you get

- Research briefs with sections: Summary, Key Findings, Sources, Open Questions
- Reliability-scored sources with short rationale
- Multi-tool evidence gathering
- Local memory in `data/<project>/research_notes.json`
- Download buttons for `.md` and `.pdf` brief exports

## Suggested extensions

1. Add citation deduplication and URL normalization.
2. Add source freshness checks (publication date scoring).
3. Add per-brief feedback and rerun controls.
