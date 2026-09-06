"""
RAG Engine
- Embeds text chunks with fastembed (ONNX Runtime, CPU-only, no PyTorch)
- Stores/searches with FAISS
- Answers with a free, fully local Gemma model served through Ollama
  (lightweight, quantized GGUF, no PyTorch/transformers install needed),
  or falls back to an extractive answer with no LLM at all.
"""

import os
import json
import asyncio
import threading
from pathlib import Path
from typing import List, AsyncIterator, Optional

import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────────
# fastembed model name (ONNX-based, no PyTorch dependency — avoids Windows
# Smart App Control / WDAC blocking torch's native DLLs). BAAI/bge-small-en-v1.5
# is a small, accurate, CPU-friendly embedding model (~130MB).
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")   # "ollama" | "gemma" | "openai" | "extractive"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "5"))

# Minimum cosine-similarity score a retrieved chunk needs before we let the
# LLM attempt an answer at all. If the best match is below this, the
# question isn't actually covered by the documents, and asking the LLM to
# answer anyway is the single most common cause of hallucination in RAG
# apps — the model politely "fills in" from its own training knowledge
# instead of admitting the documents don't cover it.
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))

# ── Ollama settings (recommended: free, fully local, no torch/transformers) ───
# Runs Gemma as a quantized GGUF model via a lightweight local server
# (llama.cpp under the hood). No PyTorch install (~2GB+ saved), much lower
# RAM use than the HuggingFace transformers path below, and doesn't touch
# Windows Smart App Control since it's a signed standalone app.
# Setup: install https://ollama.com/download, then run `ollama pull gemma2:2b`
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")

# ── Local Gemma settings ──────────────────────────────────────────────────────
# Free, runs 100% locally via Hugging Face `transformers`. No API key, no
# per-token cost. Gated on HuggingFace — you must accept the license once at
# https://huggingface.co/google/gemma-2-2b-it and set HF_TOKEN in .env.
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "google/gemma-2-2b-it")
HF_TOKEN = os.getenv("HF_TOKEN", "") or None
# "auto" = use a GPU if one is available, otherwise CPU. Set to "cpu" to force
# CPU even if a GPU is present.
GEMMA_DEVICE = os.getenv("GEMMA_DEVICE", "auto")
# "auto" = apply CPU dynamic int8 quantization automatically when running on
# CPU (roughly 2-3x faster generation, small quality tradeoff). Set to
# "none" to disable, or "int8" to force it (including on GPU, not recommended).
GEMMA_QUANTIZE = os.getenv("GEMMA_QUANTIZE", "auto")
GEMMA_MAX_NEW_TOKENS = int(os.getenv("GEMMA_MAX_NEW_TOKENS", "512"))


# ── Embedder ────────────────────────────────────────────────────────────────────
class Embedder:
    """
    Wraps fastembed.TextEmbedding — runs on ONNX Runtime (CPU), no PyTorch.
    This avoids Windows Smart App Control / WDAC blocking torch's native DLLs
    (torch_cpu.dll, shm.dll), which are unsigned at the Enterprise level.
    Extraction/embedding stays on this fast ONNX path regardless of which
    LLM_BACKEND is chosen, so document indexing is always quick.
    """

    _instance = None

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        from fastembed import TextEmbedding
        print(f"[RAG] Loading embedding model '{EMBED_MODEL}' (ONNX, CPU) …")
        self.model = TextEmbedding(model_name=EMBED_MODEL)
        print("[RAG] Embedding model ready.")

    def embed(self, texts: List[str]) -> np.ndarray:
        # fastembed returns a generator of numpy arrays; normalize for cosine
        # similarity via FAISS inner-product index.
        vectors = list(self.model.embed(texts, batch_size=16))
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


# ── Local Gemma LLM (Hugging Face transformers) ───────────────────────────────
class LocalGemmaLLM:
    """
    Loads a Gemma instruction-tuned model once (singleton) and generates
    answers locally — no server, no internet call at inference time, free.

    - Auto-detects CPU vs GPU (GEMMA_DEVICE=auto), defaults to CPU-safe
      settings so this works with no GPU at all.
    - On CPU, applies dynamic int8 quantization to the Linear layers by
      default (GEMMA_QUANTIZE=auto), which noticeably speeds up generation
      without needing bitsandbytes/GPU-only quantization libraries.
    - Uses greedy decoding (do_sample=False) so answers stay deterministic
      and grounded in the retrieved context rather than creatively sampled.
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "LocalGemmaLLM":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        try:
            import importlib
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            AutoModelForCausalLM = transformers.AutoModelForCausalLM
            AutoTokenizer = transformers.AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "Local Gemma requires `torch` and `transformers`. "
                "Install them or set LLM_BACKEND=extractive."
            ) from e

        self.torch = torch

        # ── Device selection ──
        if GEMMA_DEVICE == "cpu":
            self.device = "cpu"
        elif GEMMA_DEVICE == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:  # auto
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16

        print(f"[Gemma] Loading '{GEMMA_MODEL}' on device='{self.device}' …")
        auth_kwargs = {}
        if HF_TOKEN:
            auth_kwargs["use_auth_token"] = HF_TOKEN

        self.tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL, **auth_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            GEMMA_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **auth_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()

        # ── Optional CPU int8 dynamic quantization (pure PyTorch, no GPU/
        # bitsandbytes dependency) — meaningfully faster CPU generation. ──
        do_quantize = (
            GEMMA_QUANTIZE == "int8"
            or (GEMMA_QUANTIZE == "auto" and self.device == "cpu")
        )
        if do_quantize and self.device == "cpu":
            try:
                print("[Gemma] Applying CPU int8 dynamic quantization …")
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                print("[Gemma] Quantization applied.")
            except Exception as e:
                print(f"[Gemma] Quantization skipped ({e}); using full precision.")

        print("[Gemma] Model ready.")

    def _build_inputs(self, system: str, user: str):
        # Gemma's chat template does not accept a separate "system" role, so
        # the system instructions are folded into the single user turn.
        combined = f"{system}\n\n{user}"
        messages = [{"role": "user", "content": combined}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        return input_ids

    def start_generation(self, system: str, user: str):
        """Kicks off generation in a background thread and returns a
        TextIteratorStreamer that yields decoded text chunks as they're
        produced, plus the thread (so callers can join it)."""
        import importlib

        transformers = importlib.import_module("transformers")
        TextIteratorStreamer = transformers.TextIteratorStreamer

        input_ids = self._build_inputs(system, user)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=GEMMA_MAX_NEW_TOKENS,
            do_sample=False,        # greedy — deterministic, no creative drift
            num_beams=1,
            repetition_penalty=1.1,
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        return streamer, thread


def _next_or_none(it):
    """Pull the next item from a blocking iterator, or None at the end.
    Used so we can drive the (blocking) TextIteratorStreamer from an async
    generator via a thread-pool executor, without blocking the event loop."""
    try:
        return next(it)
    except StopIteration:
        return None


# ── RAG Engine ──────────────────────────────────────────────────────────────────
class RAGEngine:
    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.chunks: List[str] = []
        self.sources: List[str] = []
        self.index = None  # FAISS index, lazy-init on first add
        self._load_state()

    # ── State persistence ──────────────────────────────────────────────────────
    def _meta_path(self):
        return self.store_path / "meta.json"

    def _index_path(self):
        return self.store_path / "faiss.index"

    def _load_state(self):
        meta = self._meta_path()
        idx = self._index_path()
        if meta.exists() and idx.exists():
            try:
                import faiss
                data = json.loads(meta.read_text())
                self.chunks = data.get("chunks", [])
                self.sources = data.get("sources", [])
                self.index = faiss.read_index(str(idx))
                print(f"[RAG] Loaded {len(self.chunks)} chunks from disk.")
            except Exception as e:
                print(f"[RAG] Could not load saved index: {e}")

    def _save_state(self):
        try:
            import faiss
            meta = {"chunks": self.chunks, "sources": self.sources}
            self._meta_path().write_text(json.dumps(meta))
            if self.index is not None:
                faiss.write_index(self.index, str(self._index_path()))
        except Exception as e:
            print(f"[RAG] Warning: could not persist index: {e}")

    # ── Indexing ───────────────────────────────────────────────────────────────
    def add_documents(self, chunks: List[str], source: str = ""):
        import faiss
        embedder = Embedder.get()
        vectors = embedder.embed(chunks)  # (N, D)

        dim = vectors.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)  # Inner-product (cosine after norm)

        self.index.add(vectors)
        self.chunks.extend(chunks)
        self.sources.extend([source] * len(chunks))
        self._save_state()
        print(f"[RAG] Indexed {len(chunks)} chunks from '{source}'.")

    def has_documents(self) -> bool:
        return bool(self.chunks) and self.index is not None and self.index.ntotal > 0

    # ── Retrieval ──────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[dict]:
        if not self.has_documents():
            return []
        embedder = Embedder.get()
        q_vec = embedder.embed([query])  # (1, D)
        scores, indices = self.index.search(q_vec, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "text": self.chunks[idx],
                "source": self.sources[idx],
                "score": float(score),
            })
        return results

    def retrieve_sources(self, query: str) -> List[dict]:
        return self.retrieve(query)

    # ── Answer generation ──────────────────────────────────────────────────────
    async def stream_answer(
        self, question: str, history: List[dict]
    ) -> AsyncIterator[str]:
        hits = self.retrieve(question)
        if not hits:
            yield "I could not find relevant information in the provided documents."
            return

        # Hallucination gate: if even the best-matching chunk is a weak
        # match, don't ask the LLM to answer at all — refuse deterministically
        # instead of letting the model improvise from a barely-related chunk.
        if hits[0]["score"] < MIN_RELEVANCE_SCORE:
            yield "I could not find this in the provided document."
            return

        # Label each chunk with a numbered source tag so the model can (and
        # is instructed to) cite exactly which excerpt it drew from — this
        # makes ungrounded claims easy to spot, since a made-up fact won't
        # map to any [Excerpt N] tag.
        context = "\n\n".join(
            f"[Excerpt {i+1} — source: {h['source']}]\n{h['text']}"
            for i, h in enumerate(hits)
        )
        history_text = ""
        for msg in history[-6:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                history_text += f"{role.capitalize()}: {content}\n"

        system_prompt = (
            "You are a careful assistant that answers questions using ONLY the "
            "numbered excerpts in the document context below. Rules:\n"
            "1. Every claim in your answer must be traceable to at least one "
            "excerpt. Do not add facts, numbers, names, or assumptions that "
            "are not explicitly stated in the excerpts.\n"
            "2. When you state a fact, mention which excerpt it came from, "
            "e.g. '(Excerpt 2)'.\n"
            "3. If the excerpts only partially answer the question, answer "
            "only the part they cover and explicitly say what is missing.\n"
            "4. If the excerpts do not contain the answer at all, reply "
            "exactly: 'I could not find this in the provided document.' Do "
            "not guess, generalize, or use outside/background knowledge to "
            "fill the gap, even if you believe you know the answer.\n"
            "5. Be concise. Do not add commentary, disclaimers, or content "
            "the user did not ask for."
        )
        user_prompt = (
            f"Document Context:\n{context}\n\n"
            + (f"Conversation so far:\n{history_text}\n" if history_text else "")
            + f"Question: {question}\n\nAnswer:"
        )

        backend = LLM_BACKEND.lower()
        if backend == "ollama":
            async for token in self._stream_ollama(system_prompt, user_prompt):
                yield token
        elif backend == "gemma":
            async for token in self._stream_gemma(system_prompt, user_prompt):
                yield token
        elif backend == "openai":
            async for token in self._stream_openai(system_prompt, user_prompt):
                yield token
        else:
            # Extractive fallback — no LLM needed at all
            async for token in self._extractive_answer(question, hits):
                yield token

    # ── Ollama streaming (recommended: lightweight, no torch/transformers) ─────
    async def _stream_ollama(self, system: str, user: str) -> AsyncIterator[str]:
        import aiohttp

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            # temperature=0 → deterministic, greedy-style decoding, same
            # anti-hallucination reasoning as the local Gemma backend: we
            # want the model to stay close to the retrieved context, not
            # sample creatively.
            "options": {"temperature": 0},
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_HOST}/api/chat", json=payload
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        yield (
                            f"\n\n[Ollama error {resp.status}]: {text}\n"
                            f"Make sure the model is pulled: `ollama pull {OLLAMA_MODEL}`"
                        )
                        return
                    async for raw_line in resp.content:
                        line = raw_line.decode().strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
        except aiohttp.ClientConnectorError:
            yield (
                "\n\n[Could not connect to Ollama]: Is it running? Start the "
                "Ollama app (or run `ollama serve`), and make sure you've "
                f"pulled the model: `ollama pull {OLLAMA_MODEL}`."
            )

    # ── Local Gemma streaming ───────────────────────────────────────────────────
    async def _stream_gemma(self, system: str, user: str) -> AsyncIterator[str]:
        try:
            llm = LocalGemmaLLM.get()
        except Exception as e:
            yield (
                f"\n\n[Could not load local Gemma model]: {e}\n"
                "Make sure `transformers`/`torch` are installed, you have "
                "accepted the Gemma license on HuggingFace, and HF_TOKEN is "
                "set in .env. You can also set LLM_BACKEND=extractive to skip "
                "the LLM entirely."
            )
            return

        loop = asyncio.get_event_loop()
        try:
            streamer, thread = await loop.run_in_executor(
                None, llm.start_generation, system, user
            )
        except Exception as e:
            yield f"\n\n[Gemma generation error]: {e}"
            return

        it = iter(streamer)
        try:
            while True:
                token = await loop.run_in_executor(None, _next_or_none, it)
                if token is None:
                    break
                yield token
        finally:
            await loop.run_in_executor(None, thread.join, 1.0)

    # ── OpenAI streaming ───────────────────────────────────────────────────────
    async def _stream_openai(self, system: str, user: str) -> AsyncIterator[str]:
        import aiohttp
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                async for line in resp.content:
                    line = line.decode().strip()
                    if line.startswith("data: "):
                        line = line[6:]
                    if line == "[DONE]" or not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = (
                            data.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue

    # ── Extractive fallback ────────────────────────────────────────────────────
    async def _extractive_answer(
        self, question: str, hits: List[dict]
    ) -> AsyncIterator[str]:
        """Simple extractive answer — no LLM needed. Returns top chunks."""
        intro = (
            "Based on the document, here is the most relevant information I found:\n\n"
        )
        for char in intro:
            yield char
            await asyncio.sleep(0.005)

        for i, hit in enumerate(hits[:3], 1):
            section = f"[Excerpt {i} — {hit['source']}]\n{hit['text']}\n\n"
            for char in section:
                yield char
                await asyncio.sleep(0.002)

        note = (
            "\n_Note: This answer uses extractive retrieval. "
            "For generative answers, set LLM_BACKEND=ollama in your .env._"
        )
        for char in note:
            yield char
            await asyncio.sleep(0.003)

    # ── Reset ──────────────────────────────────────────────────────────────────
    def reset(self):
        self.chunks = []
        self.sources = []
        self.index = None
        print("[RAG] Session reset.")