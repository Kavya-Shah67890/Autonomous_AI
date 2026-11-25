# Autonomous Multi-Agent AI Research System

A production-ready multi-agent research system built with **Python**, **LangGraph**, **Gemini API (`google-genai`)**, **Playwright**, and **Streamlit**.

## Architecture

1. **Manager Agent**: Understands user instructions, decomposes the research goal into sub-questions, and orchestrates the workflow.
2. **Web Research Agent**: Searches Google/DuckDuckGo via Playwright to discover relevant sources.
3. **Investigation Agent**: Explores discovered webpages, extracts clean content, and follows relevant links.
4. **Analysis Agent**: Compares collected information across sources, identifies consensus, contradictions, and gaps.
5. **Fact-Checking Agent**: Verifies key claims against collected evidence and assigns confidence levels.
6. **Report Agent**: Produces a structured, detailed Markdown report with inline source links and references.

## Safety & Security
- **API Key Protection**: Keys are loaded from `.env` or runtime environment variables and never logged or exposed in UI/reports.
- **Permission Gates**: Sensitive actions (e.g., visiting external/untrusted domains, saving reports to disk, or running deep multi-hop crawls) can require explicit user approval in the UI.
- **Error Handling & Logging**: Structured logging to `logs/research_system.log` and live UI log streaming.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.

## Run the UI

```bash
streamlit run app.py
```

## Run Tests

```bash
pytest
```
