import gzip
import json
import glob
import os
import time

from bs4 import BeautifulSoup
import spacy
import string
from spacy.lang.en import English
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

from datetime import datetime
import re

# =======================
# Initial Setup & File Chunking
# =======================

# Start timer to measure script execution time
start_time = time.time()

# Path to the directory containing press release data
data_dir = "../../data/Press-Releases"
# data_dir = "dai-project/sample-data" # Alternative for testing

# Retrieve chunk number from environment variable for parallel processing
chunk = int(os.environ["CHUNK"])
files_per_chunk = 5  # Number of files to process per chunk

# Get sorted list of all .jsonl.gz files to ensure deterministic processing order
all_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl.gz")))

# Compute start and end indices for files in the current chunk
start_idx = (chunk - 1) * files_per_chunk
end_idx = start_idx + files_per_chunk

# Select the files to be processed in this chunk
chunk_files = all_files[start_idx:end_idx]

print(f"CHUNK={chunk}, processing files {start_idx+1} to {end_idx}:")
print("\n".join(chunk_files))

# =======================
# Load Press Releases from Files
# =======================

json_list = []
for file_path in chunk_files:
    # Open each gzipped JSON lines file and load objects line-by-line
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            json_list.append(json.loads(line))

# End timer for file loading
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")

# Print how many press releases have been loaded
print(f"Loaded {len(json_list)} JSON objects from .jsonl.gz files.")

# =======================
# NLP and Sentiment Model Setup
# =======================

#!python -m spacy download en_core_web_sm
# Load small English spaCy model for tokenization and sentence splitting
spacy_nlp = spacy.load('en_core_web_sm')

# Load FinBERT sentiment analysis model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

# Create a HuggingFace pipeline for sentiment analysis using FinBERT
sentiment_nlp = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=512
)

# =======================
# Helper Functions
# =======================

def get_sentiment_scores(sentiment_scores):
    """
    Calculate two types of normalized sentiment scores:
    - score_including_neutrals: weighted sum over all sentences (including neutral)
    - score_polarized_only: weighted sum over only positive/negative sentences
    """
    flag_sum = 0
    num_sentences = len(sentiment_scores)
    num_polarized = 0

    for s in sentiment_scores:
        if s['label'] == 'Positive':
            flag_sum += s['score']
            num_polarized += 1
        elif s['label'] == 'Negative':
            flag_sum -= s['score']
            num_polarized += 1
        # Neutral is ignored for polarized count

    score_including_neutrals = flag_sum / num_sentences if num_sentences else 0
    score_polarized_only = flag_sum / num_polarized if num_polarized else 0

    return score_including_neutrals, score_polarized_only

# =======================
# Main Extraction Function
# =======================

def extract_data(press_release):
    """
    Extracts relevant sentiment and ticker data from a single press release JSON object.
    Returns a list of output dictionaries, one per ticker found.
    """
    # Extract and format the date as 'YYYY-MM-DD'
    date = press_release["Date"]
    date = datetime.fromisoformat(date.replace('Z', '+00:00'))
    date = date.strftime('%Y-%m-%d')

    # Extract the HTML content of the press release
    html_str = press_release["Document"]["Content"]
    text = ""
    if html_str:
        # Parse HTML using BeautifulSoup, extract text from <nitf:body.content>
        soup = BeautifulSoup(html_str, 'html.parser')
        body_tag = soup.find('nitf:body.content')
        if body_tag:
            text = body_tag.get_text()

    # Only proceed if there is text content to analyze
    if text:
        def pattern_clean(text):
            """
            Clean up bullet points and numbered patterns from the text.
            """
            # Replace bullet points with periods
            text = re.sub(r'•\s+', '. ', text)
            text = re.sub(r'•', '. ', text)
            # Remove patterns like ".#" or ".# "
            text = re.sub(r'\.\d+\s+', '. ', text)
            text = re.sub(r'\.\d+', '. ', text)
            return text
        text = pattern_clean(text)

        # Split the cleaned text into sentences using spaCy
        doc = spacy_nlp(text)
        sentences = [sent.text for sent in doc.sents]

        def is_relevant_sentence(sentence):
            """
            Filter out short and metadata-like sentences (dates, times, etc.).
            """
            metadata_patterns = [
                r"\bEastern Time\b",
                r"\b\d{4}\b",  # Year
                r"\bJanuary|\bFebruary|\bMarch|\bApril|\bMay|\bJune|\bJuly|\bAugust|\bSeptember|\bOctober|\bNovember|\bDecember\b",
                r"\b\d{1,2}:\d{2}\s*(AM|PM)?\b",  # Time
            ]
            for pattern in metadata_patterns:
                if re.search(pattern, sentence) and len(sentence.split()) < 10:
                    return False
            # Filter out very short sentences
            if len(sentence.split()) < 4:
                return False
            return True

        # Regex to match tickers like (NASDAQ: AAPL) or [NASDAQ: AAPL]
        pattern = r'[\(\[]([A-Za-z ]+): ?([A-Za-z0-9\.\-]+) *[\)\]]'
        matches = re.findall(pattern, text, re.IGNORECASE)

        # No NASDAQ/relevant_tickers filter! We output for all tickers matched.

        # Filter out irrelevant sentences for sentiment analysis
        sentences = [s for s in sentences if is_relevant_sentence(s)]
        # Analyze each sentence's sentiment using FinBERT
        sentences_sentiment = sentiment_nlp(sentences)

        # Map to store ticker symbols found in sentences and their polarity scores
        tickers_in_text = dict()
        for item in zip(sentences, sentences_sentiment):
            sentence = item[0]
            sentiment = item[1]
            # Search for tickers in the current sentence
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            if matches:
                for m in matches:
                    # Initialize the ticker's sentiment score if not present
                    if m[1] not in tickers_in_text.keys():
                        if sentiment["label"] == "Positive":
                            tickers_in_text[m[1]] = 1
                        elif sentiment["label"] == "Negative":
                            tickers_in_text[m[1]] = -1
                        else:
                            tickers_in_text[m[1]] = 0
                    else:
                        # Update the ticker's sentiment score based on this sentence
                        if sentiment["label"] == "Positive":
                            tickers_in_text[m[1]] += 1
                        elif sentiment["label"] == "Negative":
                            tickers_in_text[m[1]] -= 1

    # Calculate overall sentiment scores for the release (both diluted and pure)
    polarity_diluted, polarity_pure = get_sentiment_scores(sentences_sentiment)
    
    # Return a list of dictionaries, one per detected ticker, with their sentiment scores and date
    return [
        {
            "ticker": ticker,
            "date": date,
            "polarity_diluted": polarity_diluted,
            "polarity_pure": polarity_pure,
            "polarity_immediate": tickers_in_text[ticker]  # Number of positive minus negative mentions in ticker sentences
        } for ticker in tickers_in_text.keys()
    ]

# =======================
# Batch Processing Setup
# =======================

BATCH_SIZE = 100  # Number of output objects per saved batch file
OUTPUT_DIR = "extracted_data_all"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

# =======================
# Main Extraction Loop
# =======================

# Timer for extraction
start_time = time.time()

batch = []        # Accumulates output records for the current batch
batch_num = 0     # Batch file counter

for idx, press_release in enumerate(json_list, start=0):
    try:
        # Extract sentiment data for all tickers in this press release
        extracted = extract_data(press_release)
        for item in extracted:
            batch.append(item)
    
        # Save current batch to disk if batch size reached
        if len(batch) >= BATCH_SIZE:
            batch_file = os.path.join(OUTPUT_DIR, f"chunk_{chunk}_batch_{batch_num}.jsonl")
            with open(batch_file, "w") as f:
                for obj in batch:
                    f.write(json.dumps(obj) + "\n")
                    f.flush()  # Ensure data is written
            elapsed = time.time() - start_time
            print(f"💾 Saved batch {batch_num} ({len(batch)} items) — elapsed time: {elapsed:.2f} sec", flush=True)
            batch = []  # Reset batch
            batch_num += 1

        # Print progress every 100 iterations
        if idx % 100 == 0:
            print(f"Iteration: {idx}, Batch Size {len(batch)}")

    except Exception as e:
        # Print errors but continue processing the next press release
        print(f"⚠️ Error processing index {idx}: {e}", flush=True)
        continue

# Save any remaining data in the last batch
if batch:
    batch_file = os.path.join(OUTPUT_DIR, f"batch_{batch_num}.jsonl")
    with open(batch_file, "w") as f:
        for obj in batch:
            f.write(json.dumps(obj) + "\n")
    elapsed = time.time() - start_time
    print(f"💾 Saved final batch {batch_num} ({len(batch)} items) — elapsed time: {elapsed:.2f} sec")

print("✅ Extraction completed.")