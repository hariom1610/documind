import logging
import re
import string
import hashlib
import threading
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_summarizer = None
_ner_pipeline = None
_sentiment_pipeline = None

_models_loading = False          # guards against double-start
_model_ready = threading.Event() # signals that loading finished (success or fail)
_lock = threading.Lock()         # must be defined before any function uses it

# ---------------------------------------------------------------------------
# In-memory result cache
# ---------------------------------------------------------------------------
_analysis_cache: dict = {}
_CACHE_MAX_SIZE = 100


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_models_bg() -> None:
    """
    Load all three HuggingFace pipelines in a background thread.
    Always calls _model_ready.set() at the end so callers are never stuck.
    """
    global _summarizer, _ner_pipeline, _sentiment_pipeline
    logger.info("Loading HuggingFace models in background — this may take a minute...")

    try:
        from transformers import pipeline

        try:
            # pyrefly: ignore [no-matching-overload]
            _summarizer = pipeline(
                task="summarization",
                model="sshleifer/distilbart-cnn-12-6",
            )
            logger.info("Summarizer loaded.")
        except Exception as e:
            logger.warning(f"Summarizer failed to load: {e}")
            _summarizer = None

        try:
            _ner_pipeline = pipeline(
                task="token-classification",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
            )
            logger.info("NER pipeline loaded.")
        except Exception as e:
            logger.warning(f"NER pipeline failed to load: {e}")
            _ner_pipeline = None

        try:
            _sentiment_pipeline = pipeline(
                task="text-classification",
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            )
            logger.info("Sentiment pipeline loaded.")
        except Exception as e:
            logger.warning(f"Sentiment pipeline failed to load: {e}")
            _sentiment_pipeline = None

        logger.info("Model loading complete.")

    except Exception as e:
        logger.error(f"Critical error during model loading: {e}")

    finally:
        # ALWAYS release waiters — even if everything failed.
        # Without this, _get_pipelines() would block for 120 s on every request.
        _model_ready.set()


def _get_pipelines():
    """
    Ensure models are loading, then block until ready (max 120 s).
    Returns the three pipelines (any of which may be None on load failure).
    """
    global _models_loading

    # Start background thread exactly once
    with _lock:
        if not _models_loading:
            _models_loading = True
            threading.Thread(target=_load_models_bg, daemon=True).start()

    # Block the calling thread until _load_models_bg signals readiness
    loaded = _model_ready.wait(timeout=120)
    if not loaded:
        raise RuntimeError(
            "AI models did not finish loading within 120 seconds. "
            "Check available RAM / disk space."
        )

    return _summarizer, _ner_pipeline, _sentiment_pipeline


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _split_into_sentence_chunks(text: str, max_chars: int = 400) -> list:
    """
    Split text on sentence boundaries so NER never receives a word split
    across two chunks (which would break entity detection).
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) < max_chars:
            current += " " + sentence
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _fallback_extractive_summary(text: str) -> str:
    """
    Pure-Python TF-IDF-style extractive summarizer used when the
    HuggingFace summarizer is unavailable or the input is tiny.
    Returns the two highest-scoring sentences in their original order.
    """
    sentences = [
        s.strip()
        for s in text.replace('\n', '. ').split('.')
        if len(s.strip()) > 10
    ]
    if len(sentences) <= 2:
        return text.strip()

    stopwords = {
        'the', 'is', 'in', 'and', 'to', 'of', 'it', 'a',
        'for', 'on', 'with', 'as', 'by', 'this', 'that',
    }
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    freq: dict = {}
    for w in words:
        if w not in stopwords and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1

    scores = []
    for i, s in enumerate(sentences):
        score = sum(
            freq.get(w, 0)
            for w in s.lower()
                       .translate(str.maketrans('', '', string.punctuation))
                       .split()
        )
        scores.append((score, i, s))

    scores.sort(reverse=True, key=lambda x: x[0])
    top = sorted(scores[:2], key=lambda x: x[1])
    return ". ".join(t[2] for t in top) + "."


def _summarize_long_text(text: str, summarizer) -> str:
    """
    Map-reduce summarization:
      1. Split text into ~800-word chunks (covers up to 6 000 words / ~12 pages).
      2. Summarize each chunk independently.
      3. Concatenate chunk summaries; if still long, run one final summarization pass.
    Falls back to extractive summary on any pipeline failure.
    """
    words = text.split()
    chunk_size = 800
    chunks = []

    for i in range(0, min(len(words), 6000), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.split()) > 30:
            chunks.append(chunk)

    if not chunks:
        return _fallback_extractive_summary(text)

    chunk_summaries = []
    for chunk in chunks:
        try:
            result = summarizer(
                chunk, max_length=60, min_length=10, do_sample=False
            )
            chunk_summaries.append(result[0]["summary_text"])
        except Exception as e:
            logger.warning(f"Chunk summarization failed: {e}")
            chunk_summaries.append(_fallback_extractive_summary(chunk))

    combined = " ".join(chunk_summaries)

    # Second pass if combined is still long
    if len(combined.split()) > 80:
        try:
            final = summarizer(
                combined, max_length=100, min_length=20, do_sample=False
            )
            return final[0]["summary_text"]
        except Exception as e:
            logger.warning(f"Final summarization pass failed: {e}")

    return combined


# ---------------------------------------------------------------------------
# Regex extractors
# ---------------------------------------------------------------------------
def _extract_amounts(text: str) -> list:
    """
    Extract monetary amounts including Indian formats (lakh/crore, Rs.).
    BERT NER does not handle currencies natively, so regex is the right tool here.
    """
    pattern = r"""(?x)
        (?:[\$\€\£\¥\₹\₩]\s?[\d,]+(?:\.\d+)?)        |
        (?:[\d,]+(?:\.\d+)?\s?(?:USD|EUR|GBP|INR|JPY)) |
        (?:(?:Rs|INR)\.?\s?[\d,]+(?:\.\d+)?)           |
        (?:[\d,]+(?:\.\d+)?\s?(?:lakh|lakhs|crore|crores))
    """
    matches = re.findall(pattern, text, re.IGNORECASE)
    return list({m.strip() for m in matches if m.strip()})


def _extract_dates(text: str) -> list:
    """
    Extract dates in common formats:
      DD/MM/YYYY, YYYY-MM-DD, 12 March 2024, March 12, 2024
    """
    pattern = r"""(?x)
        \b(?:
            \d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}                       |
            \d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}                          |
            \d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|
                         Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}           |
            (?:Jan|Feb|Mar|Apr|May|Jun|
               Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}
        )\b
    """
    return list(set(re.findall(pattern, text, re.IGNORECASE | re.VERBOSE)))


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------
def analyze_document(text: str, file_name: str) -> Dict[str, Any]:
    """
    Full analysis pipeline:
      1. Cache check   — return immediately if this exact text was seen before.
      2. Summarization — map-reduce with HuggingFace; falls back to extractive.
      3. Sentiment     — DistilBERT classifier; falls back to Neutral.
      4. NER           — BERT-NER on sentence-boundary chunks; extracts PER/ORG/LOC.
      5. Regex         — Amounts and dates (not handled well by NER models).
      6. Cache store   — evict oldest entry if cache is full.
    """
    if not text or not text.strip():
        raise ValueError("Cannot analyze empty document content.")

    # --- 0. Cache check ---
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in _analysis_cache:
        logger.info(f"Cache hit for '{file_name}'")
        return _analysis_cache[cache_key]

    logger.info(f"Analyzing '{file_name}' ({len(text)} chars) ...")
    summarizer, ner, sentiment_classifier = _get_pipelines()

    # --- 1. Summarization ---
    summary_text = ""
    try:
        word_count = len(text.split())
        if word_count <= 30:
            summary_text = "Document is too short for a reliable summary."
        elif summarizer:
            summary_text = _summarize_long_text(text, summarizer)
        else:
            summary_text = _fallback_extractive_summary(text[:2500])
    except Exception as e:
        logger.warning(f"Summarization error: {e}")
        summary_text = _fallback_extractive_summary(text[:2500])

    # --- 2. Sentiment ---
    # Use first 2 000 chars; sentiment of an entire 50-page doc is rarely meaningful
    # beyond the opening sections, and the model has a 512-token hard limit anyway.
    sentiment = "Neutral"
    try:
        if sentiment_classifier:
            result = sentiment_classifier(text[:2000])[0]
            label = result["label"].upper()
            if label == "POSITIVE":
                sentiment = "Positive"
            elif label == "NEGATIVE":
                sentiment = "Negative"
    except Exception as e:
        logger.warning(f"Sentiment error: {e}")

    # --- 3. NER ---
    names: set = set()
    organizations: set = set()
    locations: set = set()

    if ner:
        for chunk in _split_into_sentence_chunks(text, max_chars=400):
            if not chunk.strip():
                continue
            try:
                for ent in ner(chunk):
                    group = ent.get("entity_group", "")
                    word = ent.get("word", "").strip()
                    # Skip BPE continuation tokens and single characters
                    if word.startswith("##") or len(word) < 2:
                        continue
                    if group == "PER":
                        names.add(word)
                    elif group == "ORG":
                        organizations.add(word)
                    elif group == "LOC":
                        locations.add(word)
            except Exception as e:
                logger.warning(f"NER chunk error: {e}")

    # --- 4. Regex fallbacks ---
    amounts = _extract_amounts(text)
    dates = _extract_dates(text)

    # --- 5. Compile result ---
    result: Dict[str, Any] = {
        "summary": summary_text.strip(),
        "entities": {
            "names": sorted(names),
            "dates": dates,
            "organizations": sorted(organizations),
            "amounts": amounts,
            "locations": sorted(locations),
        },
        "sentiment": sentiment,
    }

    logger.info(
        f"Done — '{file_name}' | sentiment={sentiment} | "
        f"names={len(names)} orgs={len(organizations)} "
        f"locs={len(locations)} dates={len(dates)} amounts={len(amounts)}"
    )

    # --- 6. Store in cache (evict oldest if full) ---
    if len(_analysis_cache) >= _CACHE_MAX_SIZE:
        oldest = next(iter(_analysis_cache))
        del _analysis_cache[oldest]
    _analysis_cache[cache_key] = result

    return result