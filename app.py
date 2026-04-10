import os
import io
import zipfile
import re
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF

from langchain_community.document_loaders import YoutubeLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="YT → Article & PDF",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 YouTube → Article & PDF Generator")
st.caption("Paste a YouTube URL → Get article + PDF + webpage")

# ── Inputs ──────────────────────────────────────────────────
api_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
youtube_url = st.text_input("🔗 YouTube URL")
run_btn = st.button("⚡ Generate Article", use_container_width=True)


# ── Chains ──────────────────────────────────────────────────
def get_chains(key):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=key,
        temperature=0.7,
    )

    article_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a professional technical article writer."
        ),
        HumanMessagePromptTemplate.from_template("""
Convert transcript into a high-quality article.

Rules:
- Remove filler words
- Ignore sponsors / intro lines
- Focus on insights + technical value
- Add headings, bullet points, examples

Transcript:
{transcript}
""")
    ])

    webpage_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""
Generate production-ready frontend code.

FORMAT:
--html--
...
--html--

--css--
...
--css--

--js--
...
--js--
"""),
        HumanMessagePromptTemplate.from_template("""
Create a responsive article webpage.

CONTENT:
{article_content}
""")
    ])

    def load_transcript(url):
        try:
            docs = YoutubeLoader.from_youtube_url(url, add_video_info=False).load()
            if not docs:
                raise Exception("No transcript found")
            return docs[0].page_content
        except Exception as e:
            raise Exception(f"Transcript Error: {str(e)}")

    summarizer = (
        RunnableLambda(load_transcript)
        | RunnableLambda(lambda t: {"transcript": t})
        | article_prompt
        | llm
        | StrOutputParser()
    )

    webpage = (
        RunnableLambda(lambda a: {"article_content": a})
        | webpage_prompt
        | llm
        | StrOutputParser()
    )

    return summarizer, webpage


# ── Parse HTML ──────────────────────────────────────────────
def parse_output(raw):
    def extract(tag):
        try:
            return raw.split(f"--{tag}--")[1].strip()
        except:
            return ""
    return extract("html"), extract("css"), extract("js")


# ── PDF Generator (Stable Version) ──────────────────────────
def generate_pdf(article_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "YouTube Article", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=10)

    clean = re.sub(r"```.*?```", "", article_text, flags=re.DOTALL)
    clean = re.sub(r"[#*`]", "", clean)

    for line in clean.split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue

        try:
            line = line.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 6, line)
        except:
            continue

    return bytes(pdf.output())


# ── Validation ──────────────────────────────────────────────
def is_valid_url(url):
    return "youtube.com" in url or "youtu.be" in url


# ── RUN ─────────────────────────────────────────────────────
if run_btn:

    if not api_key:
        st.error("❌ Enter Groq API Key")
        st.stop()

    if not youtube_url or not is_valid_url(youtube_url):
        st.error("❌ Enter valid YouTube URL")
        st.stop()

    try:
        summarizer, webpage_chain = get_chains(api_key)

        with st.spinner("🧠 Generating article..."):
            article = summarizer.invoke(youtube_url)

        st.success("✅ Article Ready")

        # Article
        st.subheader("📄 Article")
        st.markdown(article)

        # PDF
        pdf_bytes = generate_pdf(article)
        st.download_button("📥 Download PDF", pdf_bytes, "article.pdf")

        # Webpage
        with st.spinner("🎨 Generating webpage..."):
            raw_web = webpage_chain.invoke(article)

        html, css, js = parse_output(raw_web)

        full_html = f"""
        <html>
        <head><style>{css}</style></head>
        <body>{html}<script>{js}</script></body>
        </html>
        """

        st.subheader("🌐 Preview")
        st.components.v1.html(full_html, height=600)

        # ZIP Download
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("index.html", html)
            z.writestr("style.css", css)
            z.writestr("script.js", js)
        buffer.seek(0)

        st.download_button("⬇️ Download Website", buffer, "website.zip")

    except Exception as e:
        st.error(f"❌ {str(e)}")