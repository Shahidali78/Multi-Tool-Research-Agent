import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from fpdf.errors import FPDFException
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

DATA_DIR = Path("data")
ACTIVE_PROJECT = "default"


def slugify_project(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "default"


def set_active_project(name: str) -> None:
    global ACTIVE_PROJECT
    ACTIVE_PROJECT = slugify_project(name)


def notes_path(project: Optional[str] = None) -> Path:
    project_slug = project or ACTIVE_PROJECT
    return DATA_DIR / project_slug / "research_notes.json"


def ensure_data_paths(project: Optional[str] = None) -> None:
    path = notes_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]", encoding="utf-8")


def read_notes(project: Optional[str] = None) -> List[dict]:
    ensure_data_paths(project)
    path = notes_path(project)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def append_note(topic: str, content: str, project: Optional[str] = None) -> str:
    notes = read_notes(project)
    path = notes_path(project)
    item = {
        "topic": topic,
        "content": content,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    notes.append(item)
    path.write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return "Note saved."


def search_notes(topic: str, project: Optional[str] = None) -> str:
    notes = read_notes(project)
    matches = [n for n in notes if topic.lower() in n.get("topic", "").lower()]
    if not matches:
        return f"No notes found for topic: {topic}"
    lines = []
    for n in matches[-5:]:
        lines.append(f"- [{n['created_at']}] {n['topic']}: {n['content']}")
    return "\n".join(lines)


wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1800)
)
arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=3, load_max_docs=3, load_all_available_meta=True)
)
web_tool = DuckDuckGoSearchRun()


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for background context and definitions."""
    return wiki_tool.run(query)


@tool
def arxiv_search(query: str) -> str:
    """Search arXiv for academic papers and technical findings."""
    return arxiv_tool.run(query)


@tool
def web_search(query: str) -> str:
    """Search the web for recent information and practical resources."""
    return web_tool.run(query)


@tool
def save_research_note(note: str) -> str:
    """Save an important insight as a note. Format: '<topic>: <note>'."""
    if ":" in note:
        topic, body = note.split(":", 1)
        return append_note(topic.strip(), body.strip())
    return append_note("general", note)


@tool
def read_research_notes(topic: str) -> str:
    """Read saved notes for a topic from local memory."""
    return search_notes(topic)


def model_candidates() -> List[str]:
    configured = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    fallbacks = ["gpt-4o-mini", "gpt-4.1-mini"]
    return [configured] + [m for m in fallbacks if m != configured]


def build_agent(model_name: str) -> AgentExecutor:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Add it to your .env file.")

    llm = ChatOpenAI(model=model_name, temperature=0)
    tools = [
        wikipedia_search,
        arxiv_search,
        web_search,
        save_research_note,
        read_research_notes,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a research assistant. Use tools to gather evidence before answering. "
                "Always return a concise brief with these sections: "
                "Summary, Key Findings, Sources, and Open Questions. "
                "In Sources, provide a reliability tag for each item: [High], [Medium], or [Low], "
                "followed by a very short reason. "
                "Include concrete source names or links in Sources. "
                "If any tool fails, continue with remaining tools and explicitly note the gap. "
                "If notes exist, use them to avoid repeating work.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=8)


def run_research_with_fallback(query: str) -> Tuple[str, str]:
    last_error = None
    for model_name in model_candidates():
        try:
            agent = build_agent(model_name)
            result = agent.invoke({"input": query})
            return result.get("output", "No output generated."), model_name
        except Exception as exc:
            last_error = exc
            error_text = str(exc).lower()
            if "model" in error_text and ("not" in error_text or "unavailable" in error_text):
                continue
            raise
    raise RuntimeError(
        f"Failed to run with configured model and fallbacks. Last error: {last_error}"
    )


def brief_markdown(project: str, topic: str, goal: str, output: str, model: str) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    return (
        f"# Research Brief\n\n"
        f"- Project: {project}\n"
        f"- Topic: {topic}\n"
        f"- Goal: {goal}\n"
        f"- Model: {model}\n"
        f"- Generated at: {now}\n\n"
        f"{output}\n"
    )


def markdown_to_pdf_bytes(markdown_text: str) -> bytes:
    def normalize_for_pdf(line: str) -> str:
        line = line.replace("\u200b", "").replace("\ufeff", "")
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", line)
        line = line.replace("**", "").replace("__", "").replace("`", "")
        line = line.replace("•", "-").replace("â€¢", "-")
        return re.sub(r"\s+", " ", line).strip()

    def break_long_tokens(line: str, max_token_len: int = 60) -> str:
        tokens = line.split(" ")
        fixed = []
        for token in tokens:
            if len(token) <= max_token_len:
                fixed.append(token)
                continue
            chunks = [token[i : i + max_token_len] for i in range(0, len(token), max_token_len)]
            fixed.append(" ".join(chunks))
        return " ".join(fixed)

    def classify_line(line: str) -> Tuple[str, str]:
        if not line:
            return "blank", ""
        if line.startswith("### "):
            return "h3", line[4:].strip()
        if line.startswith("## "):
            return "h2", line[3:].strip()
        if line.startswith("# "):
            return "h1", line[2:].strip()
        if re.match(r"^\d+\.\s+", line):
            return "num", line
        if line.startswith("- ") or line.startswith("* "):
            return "bullet", line[2:].strip()
        return "text", line

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    usable_width = max(10, pdf.w - pdf.l_margin - pdf.r_margin)

    def write_line(line: str, font_size: int = 11, style: str = "") -> None:
        pdf.set_font("Arial", style=style, size=font_size)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_width, 6, txt=line or " ")

    for raw_line in markdown_text.splitlines():
        safe = normalize_for_pdf(raw_line).encode("latin-1", "replace").decode("latin-1")
        safe = break_long_tokens(safe)
        kind, content = classify_line(safe)

        if kind == "blank":
            try:
                write_line(" ")
            except FPDFException:
                pass
            continue

        font_size = 11
        style = ""
        if kind == "h1":
            content = content.upper()
            font_size = 16
            style = "B"
        elif kind == "h2":
            font_size = 14
            style = "B"
        elif kind == "h3":
            font_size = 12
            style = "B"
        elif kind == "bullet":
            content = "- " + content

        try:
            write_line(content, font_size=font_size, style=style)
        except FPDFException:
            hard_wrapped = [content[i : i + 80] for i in range(0, len(content), 80)] or [" "]
            for chunk in hard_wrapped:
                try:
                    write_line(chunk, font_size=font_size, style=style)
                except FPDFException:
                    for ch in chunk:
                        try:
                            write_line(ch, font_size=font_size, style=style)
                        except FPDFException:
                            continue

    payload = pdf.output(dest="S")
    if isinstance(payload, str):
        return payload.encode("latin-1", errors="replace")
    return bytes(payload)


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "project" not in st.session_state:
        st.session_state.project = "default"


def main() -> None:
    st.set_page_config(page_title="Multi-Tool Research Agent", page_icon=":mag:")
    st.title("Multi-Tool Personal Research Agent")
    st.caption("LangChain agent using web, arXiv, Wikipedia, and local research notes")

    init_state()

    with st.sidebar:
        st.subheader("Research Input")
        project = st.text_input("Project", value=st.session_state.project)
        st.session_state.project = project
        set_active_project(project)
        topic = st.text_input("Topic", placeholder="e.g., retrieval-augmented generation evaluation")
        goal = st.text_area(
            "Goal",
            placeholder="What exactly should the agent deliver?",
            height=120,
        )
        depth = st.selectbox("Depth", ["quick", "standard", "deep"], index=1)
        run = st.button("Run Research", type="primary")

    if run:
        if not topic.strip() or not goal.strip():
            st.warning("Provide both topic and goal.")
            return

        with st.spinner("Researching across tools..."):
            query = (
                f"Topic: {topic}\n"
                f"Goal: {goal}\n"
                f"Depth: {depth}\n"
                "Use at least two external tools and check local notes. "
                "End with an actionable brief."
            )
            try:
                output, used_model = run_research_with_fallback(query)
            except Exception as exc:
                st.error(
                    "Research failed. Check OPENAI_API_KEY, model availability, and network access. "
                    f"Details: {exc}"
                )
                return

        st.session_state.history.append(
            {
                "project": slugify_project(project),
                "topic": topic,
                "goal": goal,
                "output": output,
                "model": used_model,
            }
        )

    if st.session_state.history:
        st.subheader("Latest Brief")
        latest = st.session_state.history[-1]
        st.caption(
            f"Project: `{latest['project']}` | Model: `{latest['model']}` | "
            "Please verify critical claims in the cited sources."
        )
        st.markdown(latest["output"])

        if st.button("Save Latest Brief As Note"):
            append_note(latest["topic"], latest["output"], latest["project"])
            st.success("Saved to local notes.")

        md_content = brief_markdown(
            latest["project"],
            latest["topic"],
            latest["goal"],
            latest["output"],
            latest["model"],
        )
        st.download_button(
            "Download Brief (.md)",
            data=md_content,
            file_name=f"brief_{latest['project']}_{slugify_project(latest['topic'])}.md",
            mime="text/markdown",
        )
        try:
            pdf_data = markdown_to_pdf_bytes(md_content)
            st.download_button(
                "Download Brief (.pdf)",
                data=pdf_data,
                file_name=f"brief_{latest['project']}_{slugify_project(latest['topic'])}.pdf",
                mime="application/pdf",
            )
        except Exception as exc:
            st.warning(f"PDF export unavailable for this brief. Error: {exc}")

        with st.expander("Past Briefs"):
            for idx, item in enumerate(reversed(st.session_state.history), start=1):
                st.markdown(f"### Brief {idx}: {item['topic']} ({item['project']})")
                st.markdown(item["output"])


if __name__ == "__main__":
    main()
