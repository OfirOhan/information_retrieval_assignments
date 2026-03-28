import re
import json
import time
import random
import requests
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Option 4: corpus for stop-list generation
NUM_DOCS_STOPLIST = 60

# Option 3: corpus for Boolean retrieval
NUM_DOCS_RETRIEVAL = 20

# DF-based stop-word rule
DF_RATIO_THRESHOLD = 0.25
MIN_TOTAL_FREQ = 20

# Minimum token length
MIN_TOKEN_LENGTH = 2

# Remove obvious Wikipedia artifact words
WIKI_ARTIFACTS = {"references", "external", "links", "category"}

# Fetching
MAX_FETCH_ATTEMPTS = 400
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

# Replace with your real email if you want
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "IRAssignmentBot/1.0 (student IR assignment; contact: your_email@example.com)",
    "Accept": "application/json"
})

# Files
BASE_DIR = Path("wiki_ir_project")
STOPLIST_DIR = BASE_DIR / "stoplist_docs"
RETRIEVAL_DIR = BASE_DIR / "retrieval_docs"
OUTPUT_DIR = BASE_DIR / "outputs"

for folder in [STOPLIST_DIR, RETRIEVAL_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Text preprocessing
# ------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """
    Lowercase and keep alphabetic tokens only.
    Remove 1-letter junk tokens.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return [t for t in tokens if len(t) >= MIN_TOKEN_LENGTH]


def preprocess_text(text: str, stopwords: set[str] | None = None) -> list[str]:
    """
    Tokenize, remove Wikipedia artifacts,
    and optionally remove stop words.
    """
    tokens = tokenize(text)
    tokens = [t for t in tokens if t not in WIKI_ARTIFACTS]

    if stopwords is not None:
        tokens = [t for t in tokens if t not in stopwords]

    return tokens


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# ------------------------------------------------------------
# 3. Wikipedia API helpers
# ------------------------------------------------------------
def get_random_wikipedia_title() -> str:
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": 1
    }
    response = SESSION.get(WIKI_API_URL, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    return data["query"]["random"][0]["title"]


def get_page_extract(title: str) -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": title,
        "explaintext": 1,
        "redirects": 1
    }
    response = SESSION.get(WIKI_API_URL, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    pages = data["query"]["pages"]
    page_data = next(iter(pages.values()))
    return page_data.get("extract", "")


def clean_wikipedia_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_random_wikipedia_documents(
    num_docs: int,
    save_dir: Path
) -> list[dict]:
    """
    Download random Wikipedia pages and save full plain-text extracts.
    Returns:
        [
            {
                "doc_id": "...",
                "title": "...",
                "text": "..."
            }
        ]
    """
    documents = []
    seen_titles = set()
    attempts = 0

    while len(documents) < num_docs and attempts < MAX_FETCH_ATTEMPTS:
        attempts += 1

        try:
            title = get_random_wikipedia_title()
            if title in seen_titles:
                continue

            raw_text = get_page_extract(title)
            raw_text = clean_wikipedia_text(raw_text)

            if not raw_text:
                continue

            words = raw_text.split()
            if len(words) < 50:
                continue

            full_text = " ".join(words)

            doc_id = f"D{len(documents) + 1:02d}"
            doc = {
                "doc_id": doc_id,
                "title": title,
                "text": full_text
            }

            documents.append(doc)
            seen_titles.add(title)

            safe_title = re.sub(r"[^a-zA-Z0-9_\-]+", "_", title)[:80]
            save_text(save_dir / f"{doc_id}_{safe_title}.txt", full_text)

            print(f"Fetched {doc_id}: {title}")
            time.sleep(0.2)

        except Exception as e:
            print(f"Skipping page due to error: {e}")

    return documents


# ------------------------------------------------------------
# 4. Corpus statistics and DF-based stop-list
# ------------------------------------------------------------
def compute_corpus_statistics(documents: list[dict]):
    """
    Returns:
    - total_freq: term -> total count in corpus
    - doc_freq: term -> number of documents containing the term
    - num_docs: number of documents
    """
    total_freq = Counter()
    doc_freq = Counter()

    for doc in documents:
        tokens = preprocess_text(doc["text"])
        total_freq.update(tokens)
        doc_freq.update(set(tokens))

    return total_freq, doc_freq, len(documents)


def build_df_based_stoplist(
    documents: list[dict],
    df_ratio_threshold: float = 0.25,
    min_total_freq: int = 20
):
    total_freq, doc_freq, num_docs = compute_corpus_statistics(documents)

    rows = []
    stopwords = []

    for term in doc_freq:
        tf = total_freq[term]
        df = doc_freq[term]
        df_ratio = df / num_docs
        is_stopword = (df_ratio >= df_ratio_threshold) and (tf >= min_total_freq)

        rows.append({
            "term": term,
            "total_frequency": tf,
            "document_frequency": df,
            "df_ratio": round(df_ratio, 4),
            "is_stopword": is_stopword
        })

        if is_stopword:
            stopwords.append(term)

    stats_df = pd.DataFrame(rows).sort_values(
        by=["document_frequency", "total_frequency", "term"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    stopwords = sorted(stopwords)
    return stopwords, stats_df


# ------------------------------------------------------------
# 5. Option 4: generate DF-based stop-list
# ------------------------------------------------------------
print("Downloading documents for DF-based stop-list generation...")
stoplist_docs = fetch_random_wikipedia_documents(
    num_docs=NUM_DOCS_STOPLIST,
    save_dir=STOPLIST_DIR
)

print(f"\nCollected {len(stoplist_docs)} documents for stop-list generation.")

generated_stopwords, stopword_stats_df = build_df_based_stoplist(
    stoplist_docs,
    df_ratio_threshold=DF_RATIO_THRESHOLD,
    min_total_freq=MIN_TOTAL_FREQ
)

stopword_rows = stopword_stats_df[stopword_stats_df["is_stopword"]].copy()

print(f"\nGenerated {len(generated_stopwords)} DF-based stop words.")
if stopword_rows.empty:
    print("No stop words were generated.")
else:
    print(stopword_rows.to_string(index=False))

save_text(OUTPUT_DIR / "df_based_stopwords.txt", "\n".join(generated_stopwords))
stopword_stats_df.to_csv(OUTPUT_DIR / "df_stopword_stats.csv", index=False)

print("\nGenerated stop words:")
print(generated_stopwords)


# ------------------------------------------------------------
# 6. Build inverted index for Option 3
# ------------------------------------------------------------
print("\nDownloading documents for Boolean retrieval...")
retrieval_docs = fetch_random_wikipedia_documents(
    num_docs=NUM_DOCS_RETRIEVAL,
    save_dir=RETRIEVAL_DIR
)

print(f"\nCollected {len(retrieval_docs)} documents for retrieval.")


def build_inverted_index(documents: list[dict], stopwords: set[str]):
    """
    Build inverted index:
    term -> sorted list of doc_ids
    """
    index = defaultdict(set)
    doc_titles = {}

    for doc in documents:
        doc_id = doc["doc_id"]
        doc_titles[doc_id] = doc["title"]

        tokens = preprocess_text(doc["text"], stopwords=stopwords)

        for token in set(tokens):
            index[token].add(doc_id)

    final_index = {term: sorted(postings) for term, postings in index.items()}
    return final_index, doc_titles


stopword_set = set(generated_stopwords)
inverted_index, doc_titles = build_inverted_index(retrieval_docs, stopword_set)

print(f"\nIndexed vocabulary size: {len(inverted_index)}")

sample_terms = sorted(inverted_index.keys())[:20]
sample_df = pd.DataFrame({
    "term": sample_terms,
    "postings": [inverted_index[t] for t in sample_terms]
})

print("\n20 words and their postings:")
print(sample_df.to_string(index=False))


# ------------------------------------------------------------
# 7. Boolean retrieval
# ------------------------------------------------------------
ALL_DOC_IDS = {doc["doc_id"] for doc in retrieval_docs}


def postings_for_term(term: str, index: dict) -> set[str]:
    return set(index.get(term.lower(), []))


def evaluate_boolean_query(query: str, index: dict, all_doc_ids: set[str]) -> set[str]:
    """
    Supports:
    - term
    - NOT term
    - term1 AND term2
    - term1 AND NOT term2
    """
    parts = query.strip().split()

    if len(parts) == 1:
        return postings_for_term(parts[0], index)

    if len(parts) == 2 and parts[0].upper() == "NOT":
        return all_doc_ids - postings_for_term(parts[1], index)

    if len(parts) == 3 and parts[1].upper() == "AND":
        left = postings_for_term(parts[0], index)
        right = postings_for_term(parts[2], index)
        return left & right

    if len(parts) == 4 and parts[1].upper() == "AND" and parts[2].upper() == "NOT":
        left = postings_for_term(parts[0], index)
        right = postings_for_term(parts[3], index)
        return left & (all_doc_ids - right)

    raise ValueError(
        "Supported formats:\n"
        "term\n"
        "NOT term\n"
        "term1 AND term2\n"
        "term1 AND NOT term2"
    )


def results_to_df(result_doc_ids: set[str], doc_titles: dict[str, str]) -> pd.DataFrame:
    rows = [{"doc_id": doc_id, "title": doc_titles[doc_id]} for doc_id in sorted(result_doc_ids)]
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 8. Example queries
# ------------------------------------------------------------
example_queries = [
    "history AND war",
    "music AND NOT film",
    "language AND english",
    "state AND government"
]

for query in example_queries:
    try:
        result = evaluate_boolean_query(query, inverted_index, ALL_DOC_IDS)
        print(f"\nQuery: {query}")
        print(f"Matching documents: {sorted(result)}")

        result_df = results_to_df(result, doc_titles)
        if result_df.empty:
            print("No matching documents.")
        else:
            print(result_df.to_string(index=False))

    except Exception as e:
        print(f"\nQuery: {query}")
        print(f"Error: {e}")


# ------------------------------------------------------------
# 9. Save retrieval outputs
# ------------------------------------------------------------
with open(OUTPUT_DIR / "inverted_index.json", "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, indent=2)

retrieval_titles_df = pd.DataFrame(
    [{"doc_id": doc["doc_id"], "title": doc["title"]} for doc in retrieval_docs]
)
retrieval_titles_df.to_csv(OUTPUT_DIR / "retrieval_doc_titles.csv", index=False)

print("\nAll files saved to:")
print(OUTPUT_DIR.resolve())


# ------------------------------------------------------------
# 10. Helpful inspection output
# ------------------------------------------------------------
print("\nSome indexed terms you can use for queries:")
print(sorted(list(inverted_index.keys()))[:300])