"""
Document Processor
Extracts text from: PDF, DOCX, XLSX, XLS, CSV, PPTX, TXT, MD, URLs
Returns a list of text chunks ready for embedding.
"""

import re
import csv
from pathlib import Path
from typing import List


CHUNK_SIZE = 600       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks


# ── Chunking ────────────────────────────────────────────────────────────────────
def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split a long text into overlapping chunks by sentence boundary where possible."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # Try to break at a sentence/paragraph boundary
        if end < text_len:
            for boundary in ("\n", ". ", "! ", "? "):
                pos = text.rfind(boundary, start + overlap, end)
                if pos != -1:
                    end = pos + len(boundary)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Reached the end of the text — stop. Otherwise this loops forever:
        # once `end` is pinned at text_len, `end - overlap` keeps producing
        # the same `start`, an infinite loop that also exhausts memory as
        # `chunks` grows without bound (this was the cause of the 422/500
        # upload failures and process hangs).
        if end >= text_len:
            break

        new_start = end - overlap
        # Guard against a boundary match that doesn't move `start` forward
        # (e.g. sentence punctuation found right at the edge of `overlap`),
        # which would otherwise also spin forever.
        if new_start <= start:
            new_start = end
        start = new_start
    return chunks


# ── File Parsers ────────────────────────────────────────────────────────────────
class DocumentProcessor:

    def process_file(self, filepath: str, filename: str) -> List[str]:
        suffix = Path(filename).suffix.lower()
        text = ""

        if suffix == ".pdf":
            text = self._read_pdf(filepath)
        elif suffix in (".docx", ".doc"):
            text = self._read_docx(filepath)
        elif suffix in (".xlsx", ".xls"):
            text = self._read_excel(filepath)
        elif suffix == ".csv":
            text = self._read_csv(filepath)
        elif suffix in (".pptx", ".ppt"):
            text = self._read_pptx(filepath)
        elif suffix in (".txt", ".md"):
            text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        return split_text(text)

    # PDF -----------------------------------------------------------------------
    def _read_pdf(self, filepath: str) -> str:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(filepath)
            pages = []
            for page in doc:
                pages.append(page.get_text())
            return "\n".join(pages)
        except ImportError:
            pass

        # Fallback: pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except ImportError:
            raise RuntimeError(
                "No PDF library found. Install PyMuPDF: pip install pymupdf"
            )

    # DOCX ----------------------------------------------------------------------
    def _read_docx(self, filepath: str) -> str:
        import docx
        doc = docx.Document(filepath)
        parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                if row_text.strip(" |"):
                    parts.append(row_text)
        return "\n".join(parts)

    # Excel ---------------------------------------------------------------------
    def _read_excel(self, filepath: str) -> str:
        import openpyxl
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        parts = []
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            parts.append(f"[Sheet: {sheet}]")
            for row in ws.iter_rows(values_only=True):
                row_str = " | ".join(str(c) for c in row if c is not None)
                if row_str.strip():
                    parts.append(row_str)
        return "\n".join(parts)

    # CSV -----------------------------------------------------------------------
    def _read_csv(self, filepath: str) -> str:
        with open(filepath, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            rows = [" | ".join(row) for row in reader if any(c.strip() for c in row)]
        return "\n".join(rows)

    # PPTX (Upgraded to recursive text extraction) -----------------------------
    def _read_pptx(self, filepath: str) -> str:
        from pptx import Presentation
        prs = Presentation(filepath)
        parts = []

        def extract_text_from_shape(shape):
            text_parts = []
            
            # 1. Direct text box text frame
            if hasattr(shape, "text_frame") and shape.text_frame:
                if shape.text_frame.text.strip():
                    text_parts.append(shape.text_frame.text.strip())
            
            # 2. Slide Tables
            elif hasattr(shape, "has_table") and shape.has_table:
                table = shape.table
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        text_parts.append(row_text)
            
            # 3. Grouped shapes (recursive extraction)
            elif hasattr(shape, "shapes"):
                for sub_shape in shape.shapes:
                    text_parts.extend(extract_text_from_shape(sub_shape))
                    
            return text_parts

        for i, slide in enumerate(prs.slides, 1):
            parts.append(f"[Slide {i}]")
            for shape in slide.shapes:
                extracted = extract_text_from_shape(shape)
                if extracted:
                    parts.extend(extracted)
                    
        return "\n".join(parts)

    # URL -----------------------------------------------------------------------
    def process_url(self, url: str) -> List[str]:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (RAG-Chatbot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        # Prefer article/main content
        main = soup.find("article") or soup.find("main") or soup.find("body")
        text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")

        # Clean up blank lines
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 30]
        return split_text("\n".join(lines))