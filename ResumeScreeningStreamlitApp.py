# streamlit_resume_screening_app.py
import io
import os
import base64
import re
from typing import List, Tuple
import math

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest

# Pillow for image normalization (optional)
try:
    from PIL import Image
    pil_available = True
except Exception:
    pil_available = False

# PDF generation (optional)
reportlab_available = True
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:
    reportlab_available = False

# --- CONFIG (change these paths if needed) ---
MODEL_PATH = "role_classifier_ovr_lr.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"

# ===== GEMINI API KEY =====
# GEMINI_API_KEY = "" # if want to directly write gemini key in the code , write here and comment the below one
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

# Default prompt
DEFAULT_GEMINI_PROMPT = (
    "Completely study the resume and Summarize the following resume in 2 short lines such as, in starting the name of the person in resume, then years of experience (if any) and then according to you (skills, project) in short. Don not write links and email id in summary at all."
    "Keep it professional and use sentence punctuation. Avoid run-on text. Resume:\n{resume_text}\n\nSummary:"
)


# Optional: import google generative api if present.
try:
    import google.generativeai as genai
    genai_available = True
except Exception:
    genai_available = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ---- Utilities ----

@st.cache_resource
def load_model_and_vectorizer(model_path: str, vec_path: str):
    model = None
    vec = None
    le = None
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model from {model_path}: {e}")
        model = None
    try:
        vec = joblib.load(vec_path)
    except Exception as e:
        st.error(f"Failed to load vectorizer from {vec_path}: {e}")
        vec = None
    return model, vec, le


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text_pages = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text_pages.append(txt)
        combined = "\n".join(text_pages)
        return combined
    except Exception:
        return ""

def extractive_summary(text: str, num_sentences: int = 2) -> str:
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= num_sentences:
        return ' '.join(sentences).strip()
    words = re.findall(r'\w+', text.lower())
    stopwords = set(["the","and","is","in","to","of","a","for","with","on","as","by","an","be","are","this","that"])
    freq = {}
    for w in words:
        if w in stopwords: continue
        freq[w] = freq.get(w, 0) + 1
    sent_scores = {}
    for s in sentences:
        s_words = re.findall(r'\w+', s.lower())
        if not s_words: continue
        score = sum(freq.get(w, 0) for w in s_words) / len(s_words)
        sent_scores[s] = score
    best = nlargest(num_sentences, sent_scores, key=sent_scores.get)
    return ' '.join(best).strip()

def gemini_summary(text: str, api_key: str, prompt_template: str = DEFAULT_GEMINI_PROMPT) -> Tuple[bool, str]:
    """
    Extract a short summary text from Google Generative API responses.
    Handles multiple response shapes (dict, proto GenerateContentResponse, Generations, tuples/lists).
    Falls back to extractive_summary on any failure.
    """
    if not text:
        return False, ""
    if not api_key:
        # no key -> fallback
        return False, extractive_summary(text, num_sentences=2)

    # try local import (will raise if not installed)
    try:
        import google.generativeai as genai
    except Exception:
        st.warning("google.generativeai not available — using fallback summary.")
        return False, extractive_summary(text, num_sentences=2)

    # helper: try to get text from dict-like response
    def _from_dict(d):
        if not isinstance(d, dict):
            return None
        # candidate/content/parts pattern
        if "candidates" in d and isinstance(d["candidates"], list) and d["candidates"]:
            first = d["candidates"][0]
            if isinstance(first, dict) and "content" in first and isinstance(first["content"], dict):
                parts = first["content"].get("parts") or []
                if isinstance(parts, list) and parts:
                    return " ".join((p.get("text","") for p in parts if isinstance(p, dict)))
        # older shapes: output or output_text
        if "output" in d and isinstance(d["output"], str):
            return d["output"]
        if "output_text" in d and isinstance(d["output_text"], str):
            return d["output_text"]
        if "content" in d and isinstance(d["content"], dict):
            parts = d["content"].get("parts") or []
            if parts:
                return " ".join((p.get("text","") for p in parts if isinstance(p, dict)))
        # generations
        if "generations" in d and isinstance(d["generations"], list) and d["generations"]:
            g0 = d["generations"][0]
            if isinstance(g0, dict) and "text" in g0:
                return g0["text"]
        return None

    # helper: try to get text from object/proto-like
    def _from_obj(o):
        try:
            # proto: o.result.candidates[0].content.parts[*].text
            if hasattr(o, "result"):
                res = getattr(o, "result")
                # attempt protobuf -> dict conversion if available
                try:
                    from google.protobuf.json_format import MessageToDict
                    d = MessageToDict(res)
                    txt = _from_dict(d)
                    if txt:
                        return txt
                except Exception:
                    pass
                # try direct attribute access
                if hasattr(res, "candidates"):
                    cand = getattr(res, "candidates")
                    if cand and len(cand) > 0:
                        first = cand[0]
                        # content.parts
                        if hasattr(first, "content") and hasattr(first.content, "parts"):
                            parts = first.content.parts
                            texts = []
                            for p in parts:
                                if hasattr(p, "text"):
                                    texts.append(getattr(p, "text"))
                                elif isinstance(p, dict) and "text" in p:
                                    texts.append(p["text"])
                            if texts:
                                return " ".join(texts)
            # object with .generations list
            if hasattr(o, "generations"):
                gens = getattr(o, "generations")
                if gens and len(gens) > 0:
                    g0 = gens[0]
                    if hasattr(g0, "text"):
                        return getattr(g0, "text")
                    # sometimes g0 has content.parts
                    if hasattr(g0, "content") and hasattr(g0.content, "parts"):
                        parts = g0.content.parts
                        parts_text = []
                        for p in parts:
                            if hasattr(p, "text"):
                                parts_text.append(p.text)
                        if parts_text:
                            return " ".join(parts_text)
            # fallback: try simple attributes
            if hasattr(o, "output_text"):
                return getattr(o, "output_text")
            if hasattr(o, "text"):
                return getattr(o, "text")
        except Exception:
            pass
        # last resort: try parsing str() for "text": "...."
        try:
            s = str(o)
            m = re.search(r'\"text\":\s*\"([^\"]+)\"', s)
            if m:
                return m.group(1)
        except Exception:
            pass
        return None

    # build prompt and call API
    try:
        # configure
        try:
            genai.configure(api_key=api_key)
        except Exception:
            try:
                genai.api_key = api_key
            except Exception:
                pass

        prompt = prompt_template.format(resume_text=text)

        raw = None
        try:
            # try common generate API
            raw = genai.generate(model="gemini-2.0-flash", prompt=prompt)
        except Exception:
            # try alternate calls
            try:
                raw = genai.generate_text(model="text-bison-001", input=prompt)
            except Exception:
                try:
                    model_obj = genai.GenerativeModel("gemini-2.0-flash")
                    raw = model_obj.generate_content(prompt)
                except Exception:
                    raw = None

        if raw is None:
            raise RuntimeError("No response from generative API")

        # attempt to extract text from various shapes
        extracted = None

        # if tuple/list returned (some clients wrap), iterate
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, dict):
                    extracted = _from_dict(item) or extracted
                else:
                    extracted = _from_obj(item) or extracted
                if extracted:
                    break
        elif isinstance(raw, dict):
            extracted = _from_dict(raw)
        else:
            # object-like
            extracted = _from_obj(raw)

        # final fallbacks
        if not extracted:
            # maybe it's a simple object with output_text or string content
            if isinstance(raw, str):
                extracted = raw
            elif hasattr(raw, "output_text"):
                extracted = getattr(raw, "output_text")
            elif hasattr(raw, "text"):
                extracted = getattr(raw, "text")
            else:
                # try converting to dict via MessageToDict if possible
                try:
                    from google.protobuf.json_format import MessageToDict
                    d = MessageToDict(raw)
                    extracted = _from_dict(d)
                except Exception:
                    pass

        if not extracted:
            raise RuntimeError("Unable to extract text from generative response")

        # sanitize: remove newlines/emails/links and keep two sentences
        extracted = re.sub(r'\s+', ' ', str(extracted)).strip()
        extracted = re.sub(r'\S+@\S+', '', extracted)
        extracted = re.sub(r'http\S+|www\.\S+', '', extracted)
        sents = re.split(r'(?<=[.!?])\s+', extracted)
        short = ' '.join(sents[:2]).strip()
        return True, short

    except Exception as e:
        # show a helpful error in UI and fallback to extractive summary
        st.error(f"Gemini extraction failed: {e}. Using fallback summary.")
        print("Gemini extraction error:", e)
        return False, extractive_summary(text, num_sentences=2)
    
def results_to_csv_bytes(results: List[dict]) -> bytes:
    df = pd.DataFrame(results)
    return df.to_csv(index=False).encode('utf-8')


def results_to_pdf_bytes(results: List[dict], title: str = "Resume Screening Report") -> bytes:
    if not reportlab_available:
        raise RuntimeError("reportlab not available")
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x_margin = 40
    y = height - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_margin, y, title)
    y -= 30
    c.setFont("Helvetica", 10)
    for r in results:
        if y < 80:
            c.showPage()
            y = height - 60
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x_margin, y, f"File: {r.get('Resume file Name','')}")
        y -= 14
        c.setFont("Helvetica", 10)
        c.drawString(x_margin, y, f"Predicted Category: {r.get('Predicted Category','')}")
        y -= 12
        c.drawString(x_margin, y, f"Match Score: {r.get('Match Score','')}")
        y -= 12
        summary = r.get('Summary','')
        wrap_width = 90
        words = summary.split()
        line = ''
        for w in words:
            if len(line) + len(w) + 1 > wrap_width:
                c.drawString(x_margin, y, line)
                y -= 12
                line = w + ' '
            else:
                line += w + ' '
        if line:
            c.drawString(x_margin, y, line)
            y -= 14
        y -= 6
    c.save()
    buffer.seek(0)
    return buffer.read()

def render_centered_html_table(df: pd.DataFrame, column_order: List[str]):
    df = df.copy()
    df = df[column_order]
    def esc(s):
        if pd.isna(s):
            return ""
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = "<div style='overflow-x:auto;'><table style='width:100%; border-collapse: collapse;'>"
    html += "<thead><tr>"
    for col in df.columns:
        html += f"<th style='border-bottom:1px solid #ddd; padding:8px; text-align:center; font-weight:700'>{esc(col)}</th>"
    html += "</tr></thead><tbody>"
    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            v = esc(row[col])
            if col.lower().strip() == 'summary':
                html += f"<td style='padding:8px; text-align:center; vertical-align:top; white-space:normal; word-wrap:break-word; max-width:480px;'>{v}</td>"
            else:
                html += f"<td style='padding:8px; text-align:center; vertical-align:middle; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{v}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

# --- Main app ---

def main():
    st.set_page_config(page_title="Resume Screening with NLP", layout='wide')

    st.markdown(
        "<h1 style='text-align:center; margin-bottom:6px;'>Resume Screening with NLP</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center; margin-top:0; color:var(--secondary); font-size:16px;'>"
        "Upload Resumes and a Job Description — Get the Predicted Category, Match Score, and AI-generated Summary for every Resume."
        "</p>",
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        "<h2 style='text-align:center; margin-top:20px; margin-bottom:8px; font-size:50px;'>Developed By</h2>",
        unsafe_allow_html=True
    )       
    st.write("")

    def image_file_to_data_uri(path: str) -> str:
        """Convert local image path to base64 data URI (or fallback placeholder)."""
        if not path:
            return "https://via.placeholder.com/180"
        try:
            with open(path, "rb") as f:
                b = f.read()
            b64 = base64.b64encode(b).decode("ascii")
            ext = os.path.splitext(path)[1].lower().replace(".", "")
            mime = "png" if ext == "png" else ("jpeg" if ext in ("jpg","jpeg") else "png")
            return f"data:image/{mime};base64,{b64}"
        except Exception:
            return "https://via.placeholder.com/180"

    cofounders = [
        {"name": "Aryan Agarwal",  "img": "blackHoodieAryanAgarwal.png", "link": "https://www.linkedin.com/in/aryanagarwal-dev/"}
    ]

    cols = st.columns([1, 2, 1])
    for col, cf in zip([cols[1]], cofounders):
        img_uri = image_file_to_data_uri(cf.get("img", ""))
        html = f"""
        <div style="text-align:center; padding:8px;">
        <a href="{cf['link']}" target="_blank" rel="noreferrer">
            <img src="{img_uri}"
                style="width:170px; height:170px; object-fit:cover; object-position:center top;
                        border-radius:50%; display:block; margin:0 auto;
                        border:3px solid rgba(255,255,255,0.6);
                        box-shadow:0 3px 10px rgba(0,0,0,0.25);
                        transition:transform 0.25s ease;"
                onmouseover="this.style.transform='scale(1.06)'"
                onmouseout="this.style.transform='scale(1)'" />
        </a>
        <div style="margin-top:10px; font-weight:700; text-align:center;">{cf['name']}</div>
        </div>
        """
        col.markdown(html, unsafe_allow_html=True)
    st.write('---')

    # Load core model/vectorizer
    model, vectorizer, label_enc = load_model_and_vectorizer(MODEL_PATH, VECTORIZER_PATH)
    if model is None or vectorizer is None:
        st.error("Model or vectorizer not found. Please check joblib files and paths.")
        return

    if not GEMINI_API_KEY:
        st.warning("Gemini API key not found. AI summaries will use fallback extractive summarizer.")

    # --- uploader (show filenames sorted, only here) ---
    uploaded_files = st.file_uploader("Upload PDF resume files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        uploaded_files = sorted(uploaded_files, key=lambda f: f.name.lower())
        st.markdown("**Uploaded files:**")
        for f in uploaded_files:
            st.markdown(f"- {f.name}")

    # Job description input
    job_desc = None
    if uploaded_files:
        job_desc = st.text_area("Paste your Job Description here:", height=160)
        st.write("")
        process_clicked = st.button("🔍 Process Resumes")
    else:
        st.info("Upload one or more resume PDFs to enable job description input.")
        process_clicked = False

    if uploaded_files and process_clicked:
        results = []
        texts_for_vector = []
        raw_texts_for_summary = []
        extraction_failures = []

        # Extract once in sorted order
        uploaded_sorted = uploaded_files  # already sorted above
        for up in uploaded_sorted:
            fname = up.name
            bytes_data = up.read()
            raw_text = extract_text_from_pdf_bytes(bytes_data)
            cleaned = clean_text_preserve_years(raw_text) if raw_text else ""
            texts_for_vector.append(cleaned)
            raw_texts_for_summary.append(raw_text or "")
            if not cleaned:
                extraction_failures.append(fname)

        # require job desc
        if not job_desc or not job_desc.strip():
            st.error("Please Paste a Job Description before Proceeding.")
            return

        cleaned_jd = clean_text_preserve_years(job_desc)

        # vectorize
        try:
            X_resumes = vectorizer.transform(texts_for_vector)
            X_jd = vectorizer.transform([cleaned_jd])
        except Exception as e:
            st.error(f"Error transforming texts with vectorizer: {e}")
            return

        # cosine
        try:
            sims = cosine_similarity(X_resumes, X_jd).flatten()
        except Exception as e:
            st.error(f"Error computing cosine similarity: {e}")
            return

        # loop to predict and summarize
        total = len(uploaded_sorted)
        progress_bar = st.progress(0)
        for i, up in enumerate(uploaded_sorted):
            fname = up.name
            cleaned = texts_for_vector[i]
            score = float(sims[i]) if not pd.isna(sims[i]) else 0.0

            # classifier prediction
            predicted_label = ""
            try:
                if not cleaned:
                    raise ValueError("No text extracted")
                X_single = vectorizer.transform([cleaned])
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_single)[0]
                    idx = np.argmax(probs)
                    if hasattr(model, 'classes_'):
                        label = model.classes_[idx]
                    else:
                        label = str(idx)
                else:
                    label = model.predict(X_single)[0]
                # safe inverse transform
                try:
                    if label_enc is not None:
                        label = label_enc.inverse_transform([label])[0]
                except Exception:
                    pass
                predicted_label = str(label)
            except Exception:
                predicted_label = "No text extracted" if not cleaned else "Prediction Error"

            # match tag
            if score >= 0.25:
                match_tag = "High"
            elif score >= 0.1:
                match_tag = "Medium"
            else:
                match_tag = "Low"

            source_raw = (raw_texts_for_summary[i] or "")
            summary = ""
            gemini_ok, gemini_out = gemini_summary(source_raw, GEMINI_API_KEY, prompt_template=DEFAULT_GEMINI_PROMPT)
            summary = gemini_out

            results.append({
                'Resume file Name': fname,
                'Predicted Category': predicted_label,
                'Match Score': round(float(score), 4),
                'Match Tag': match_tag,
                'Summary': summary
            })

            progress_bar.progress((i + 1) / total)

        # --- sort results by Match Score (desc) and assign Rank ---
        results_sorted = sorted(results, key=lambda r: r.get('Match Score', 0), reverse=True)
        for idx, rec in enumerate(results_sorted, start=1):
            rec['Rank'] = idx

        # Create DataFrame from rank-sorted list
        df = pd.DataFrame(results_sorted)

        # Show Rank first in the table
        cols_order = ['Rank', 'Resume file Name', 'Predicted Category', 'Match Score', 'Match Tag', 'Summary']
        st.subheader("Results (ranked by Match Score)")
        render_centered_html_table(df, cols_order)

        # make download buttons larger via CSS and full-width
        st.markdown("""
        <style>
        div.stDownloadButton > button {
            width:100% !important;
            height:46px;
            font-size:15px;
        }
        </style>
        """, unsafe_allow_html=True)

        csv_bytes = results_to_csv_bytes(results)
        pdf_bytes = None
        if REPORTLAB_AVAILABLE:
            try:
                pdf_bytes = results_to_pdf_bytes(results)
            except Exception:
                pdf_bytes = None

        # place buttons side-by-side but styled to be full-width
        c1, c2 = st.columns([1,1])
        with c1:
            st.download_button("Download CSV", data=csv_bytes, file_name="ResumeScreeningResults.csv", mime='text/csv', use_container_width=True)
        with c2:
            if pdf_bytes:
                st.download_button("Download PDF", data=pdf_bytes, file_name="ResumeScreeningReport.pdf", mime='application/pdf', use_container_width=True)
            else:
                st.info("Install reportlab to enable PDF downloads: pip install reportlab")

    st.write('---')
    
def clean_text_preserve_years(text: str) -> str:
    """Placed at end to avoid forward reference; identical cleaning as above for vectorizer input."""
    if not text:
        return ""
    text = text.replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r"\s+", ' ', text)
    # remove emails/urls for vectorization as well
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r"\+?\d[\d\-\s]{6,}\d", ' ', text)
    def replace_num(m):
        s = m.group()
        if re.match(r"^(19|20)\d{2}$", s):
            return s
        return ' '
    text = re.sub(r"\b\d+\b", replace_num, text)
    text = re.sub(r"[^\w\s]", ' ', text)
    text = text.lower()
    text = re.sub(r"\s+", ' ', text).strip()
    return text

if __name__ == '__main__':
    main()