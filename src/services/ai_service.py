import logging
import re
import threading
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Lazy initialization of models so they don't block startup and are only loaded when requested.
_summarizer = None
_ner_pipeline = None
_sentiment_pipeline = None
_models_loading = False

def _load_models_bg():
    global _summarizer, _ner_pipeline, _sentiment_pipeline
    logger.info("Initializing Hugging Face Transformers models in background. This may take a minute...")
    try:
        from transformers import pipeline
        
        # Extractive / Abstractive summarizer
        try:
            _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception as e:
            logger.warning(f"Could not load summarizer: {e}")
            _summarizer = None
            
        # Token classification for Named Entity Recognition (PER, ORG, LOC, MISC)
        try:
            _ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        except Exception as e:
            logger.warning(f"Could not load NER: {e}")
            _ner_pipeline = None
            
        # Text classification for Sentiment Analysis
        try:
            _sentiment_pipeline = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            logger.warning(f"Could not load sentiment: {e}")
            _sentiment_pipeline = None
        logger.info("Transformer models successfully loaded into memory.")
    except Exception as e:
        logger.error(f"Background loading error: {e}")

def _get_pipelines():
    global _models_loading, _summarizer, _ner_pipeline, _sentiment_pipeline
    if _summarizer is None and not _models_loading:
        _models_loading = True
        threading.Thread(target=_load_models_bg, daemon=True).start()
        
    return _summarizer, _ner_pipeline, _sentiment_pipeline

def _extract_amounts(text: str) -> list[str]:
    """Fallback Regex to extract monetary amounts as BERT NER does not handle this specifically."""
    # Matches currency symbols paired with numbers like $100, $ 100, 10,000 USD, ₹100, etc.
    pattern = r"(?:[\$\€\£\¥\₹]\s?[\d,]+(?:\.\d+)?|\b[\d,]+(?:\.\d+)?\s?(?:USD|EUR|GBP|INR)\b)"
    matches = re.findall(pattern, text)
    return list(set(matches))

def _extract_dates(text: str) -> list[str]:
    """Fallback Regex to extract standard date formats like DD/MM/YYYY, YYYY-MM-DD or DD Month YYYY"""
    pattern = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b"
    return list(set(re.findall(pattern, text, re.IGNORECASE)))

def _fallback_extractive_summary(text: str) -> str:
    """A pure-Python TF-like extractive summarizer if HuggingFace summarizer fails."""
    import string
    
    # Simple sentence splitting
    sentences = [s.strip() for s in text.replace('\n', '. ').split('.') if len(s.strip()) > 10]
    if len(sentences) <= 2:
        return text.strip()
        
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    stopwords = {'the', 'is', 'in', 'and', 'to', 'of', 'it', 'a', 'for', 'on', 'with', 'as', 'by', 'this', 'that'}
    freq = {}
    
    for w in words:
        if w not in stopwords and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
            
    # Score sentences
    scores = []
    for i, s in enumerate(sentences):
        score = 0
        s_words = s.lower().translate(str.maketrans('', '', string.punctuation)).split()
        for w in s_words:
            if w in freq:
                score += freq[w]
        scores.append((score, i, s))
        
    # Get top 2 scoring sentences, sort them back to original appearance order
    scores.sort(reverse=True, key=lambda x: x[0])
    top_sentences = sorted(scores[:2], key=lambda x: x[1])
    
    return ". ".join([s[2] for s in top_sentences]) + "."

def analyze_document(text: str, file_name: str) -> Dict[str, Any]:
    """
    Main function: analyzes text using pure Hugging Face Transformer pipelines and Regex.
    No external API required.
    """
    if not text or not text.strip():
        raise ValueError("Cannot analyze empty document content")

    logger.info(f"Analyzing document locally via Transformers: {file_name} ({len(text)} chars)")

    summarizer, ner, sentiment_classifier = _get_pipelines()
    
    # 1. Summarization
    summary_text = ""
    try:
        trunc_text_summ = text[:2500]
        input_length = len(trunc_text_summ.split())
        max_len = min(60, max(20, input_length - 5))
        min_len = min(20, max_len - 5)
        
        if summarizer and input_length > 30:
            summary_result = summarizer(trunc_text_summ, max_length=max_len, min_length=min_len, do_sample=False)
            summary_text = summary_result[0]['summary_text']
        elif input_length > 30:
            # Dropdown to our safe mathematical summarizer
            summary_text = _fallback_extractive_summary(trunc_text_summ)
        else:
            summary_text = "Document is too short for a reliable summary."
    except Exception as e:
        logger.warning(f"Summarization pipeline failed: {e}")
        # Final safety net
        summary_text = _fallback_extractive_summary(text[:2500])

    # 2. Sentiment Analysis
    sentiment = "Neutral"
    try:
        if sentiment_classifier:
            trunc_text_sent = text[:2000]
            sent_result = sentiment_classifier(trunc_text_sent)[0]
            
            label = sent_result['label'].upper()
            if label == "POSITIVE":
                sentiment = "Positive"
            elif label == "NEGATIVE":
                sentiment = "Negative"
    except Exception as e:
        logger.warning(f"Sentiment pipeline failed: {e}")

    # 3. Named Entity Recognition
    names = set()
    organizations = set()
    locations = set()
    
    if ner:
        chunk_size = 1500
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if not chunk.strip(): 
                continue
            try:
                ent_results = ner(chunk)
                for ent in ent_results:
                    group = ent.get('entity_group', '')
                    word = ent.get('word', '').strip()
                    
                    if word.startswith("##") or len(word) < 2: 
                        continue
                    
                    if group == 'PER':
                        names.add(word)
                    elif group == 'ORG':
                        organizations.add(word)
                    elif group == 'LOC':
                        locations.add(word)
            except Exception as e:
                logger.warning(f"NER failed on a text chunk: {e}")

    # 4. Fallback Extraction (Dates and Amounts)
    amounts = _extract_amounts(text)
    dates = _extract_dates(text)

    # Compile result
    result = {
        "summary": summary_text.strip(),
        "entities": {
            "names": list(names),
            "dates": dates,
            "organizations": list(organizations),
            "amounts": amounts,
            "locations": list(locations)
        },
        "sentiment": sentiment
    }
    
    logger.info(
        f"Analysis complete for {file_name} | "
        f"Sentiment: {result['sentiment']} | "
        f"Names: {len(result['entities']['names'])} | "
        f"Orgs: {len(result['entities']['organizations'])}"
    )

    return result
