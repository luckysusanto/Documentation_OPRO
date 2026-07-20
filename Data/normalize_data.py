import argparse
import csv
import html
import re
import sys
import unicodedata
import emoji as _emoji # for name collision purposes
import pandas as pd
from pathlib import Path

# Placeholders
MENTION_TOKEN = "@USER"
URL_TOKEN = "HTTPURL"

# Catch URLs
RE_URL = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)
RE_URL_SPELLED = re.compile(r"\buniform resource locator\b", re.IGNORECASE)

# Catch retweet prefixes
RE_RT = re.compile(r"^\s*rt\s+@\w+\s*:\s*", re.IGNORECASE) # catch RT @USER: [TEXT]
RE_RT_TOKEN = re.compile(r"^\s*rt\s*:?\s+", re.IGNORECASE) # catch "RT ".

# Catch un-normalized handle
RE_MENTION = re.compile(r"@\w+")

# Catch CSIS placeholder "pengguna"
RE_PENGGUNA = re.compile(r"\bpengguna\b", re.IGNORECASE)

# Collapse multiple mention & URL token into one respective token
RE_MENTION_RUN = re.compile(
    r"(?:" + re.escape(MENTION_TOKEN) + r"\s*){2,}"
)
RE_URL_RUN = re.compile(
    r"(?:" + re.escape(URL_TOKEN) + r"\s*){2,}"
)

# Catch multiple whitespaces
RE_WS = re.compile(r"\s+")

# Catch leading/trailing quote/artifacts
RE_EDGE_QUOTES = re.compile(r"^['\"\s]+|['\"\s]+$")

# Catch emoji name emitted by emoji.demojize (:loudly_crying_face:)
RE_EMOJI_NAME = re.compile(r":([a-zA-Z0-9_+\-]+):")
RE_TONE_SUFFIX = re.compile(
    r"_(?:light|medium_light|medium|medium_dark|dark)_skin_tone$"
)

def _demojize_words(t: str) -> str:
    t = _emoji.demojize(t, language="en")

    def repl(m):
        name = RE_TONE_SUFFIX.sub("", m.group(1))
        name = name.split(":", 1)[0]
        return " " + name.replace("_", " ").replace("-", " " ) + " "
    
    return RE_EMOJI_NAME.sub(repl, t)

def _strip_leftover_symbols(t: str) -> str:
    out = []
    for c in t:
        cp = ord(c)
        if c == "\ufffd": # Unreadable or wrongly-rendered characters
            continue
        cat = unicodedata.category(c)
        # Drop "Symbol, other" (dingbats, misc symbols) and orphaned
        # regional indicators U+1F1E6–U+1F1FF (half-flags).
        if cat == "So" or 0x1F1E6 <= cp <= 0x1F1FF:
            continue
        out.append(c)
    return "".join(out)

def normalize(text: str, *, lowercase=False,
              keep_hashtag_hash=False) -> str:
    """Apply the full normalization pipeline to a single text string."""
    if text is None:
        return ""
    t = str(text)
 
    # 1. Structural / encoding
    t = html.unescape(t)                      # &amp; -> & , &lt; -> < ...
    t = unicodedata.normalize("NFC", t)       # composed Unicode form
 
    # 2. Retweet prefix
    t = RE_RT.sub("", t)
    t = RE_RT_TOKEN.sub("", t)
 
    # 3. URLs -> token
    t = RE_URL.sub(URL_TOKEN, t)
    t = RE_URL_SPELLED.sub(URL_TOKEN, t)
 
    # 4. Mentions -> token.  First unify CSIS 'pengguna' placeholder, then
    #    real @handles, then collapse runs.
    t = RE_PENGGUNA.sub(MENTION_TOKEN, t)     # back-map CSIS placeholder
    t = RE_MENTION.sub(MENTION_TOKEN, t)      # real @handles
    t = RE_MENTION_RUN.sub(MENTION_TOKEN + " ", t)   # collapse runs
    t = RE_URL_RUN.sub(URL_TOKEN + " ", t)           # collapse url runs too
 
    # 5. Hashtags: keep the WORD.
    if not keep_hashtag_hash:
        t = re.sub(r"#(\w+)", r"\1", t)
 
    # 6. Emojis -> English text words.
    t = _demojize_words(t)
 
    # 6b. Strip any leftover symbols the emoji library doesn't recognize
    t = _strip_leftover_symbols(t)
 
    # 7. Whitespace + edge-quote cleanup (LAST).
    t = RE_EDGE_QUOTES.sub("", t)
    t = RE_WS.sub(" ", t).strip()
 
    # 8. Casing (LAST; skip for cased transformers like IndoBERT).
    if lowercase:
        t = t.lower()
 
    return t

def read_semicolon_csv(path: Path): # Ensure data structure integrity
    df = pd.read_csv(path, sep=";", engine="python", dtype=str).fillna("")
    expected = ["id", "text", "label"]
    assert list(df.columns) == expected, (
        f"{path.name}: expected columns {expected}, got {list(df.columns)}. "
        f"A stray unquoted ';' may have added a column — inspect before "
        f"proceeding rather than dropping rows."
    )
    header = expected
    rows = [(r.id, r.text, r.label) for r in df.itertuples(index=False)]
    return header, rows



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="input CSV paths (;-delimited)")
    ap.add_argument("-o", "--outdir", default=".", help="output directory")
    ap.add_argument("--lowercase", action="store_true",
                    help="lowercase (skip for cased transformers)")
    ap.add_argument("--keep-hash", action="store_true",
                    help="keep the '#' on hashtags")
    # ap.add_argument("--add-mention-count", action="store_true",
    #                 help="add an n_mentions column (pile-on signal)")
    args = ap.parse_args()
 
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
 
    for inp in args.inputs:
        inp = Path(inp)
        header, rows = read_semicolon_csv(inp)
        out_path = outdir / (inp.stem + "_normalized.csv")
 
        n_changed = 0
        n_empty_after = 0
        with open(out_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter=";", quotechar='"',
                           quoting=csv.QUOTE_MINIMAL)
            out_header = ["id", "text", "label"]
            w.writerow(out_header)
 
            for rid, text, label in rows:
                norm = normalize(
                    text,
                    lowercase=args.lowercase,
                    keep_hashtag_hash=args.keep_hash,
                )
                if norm != str(text).strip():
                    n_changed += 1
                if not norm:
                    n_empty_after += 1
                out_row = [rid, norm, label]
                w.writerow(out_row)
 
        print(f"[{inp.name}] {len(rows)} rows -> {out_path.name} "
              f"| changed: {n_changed} | empty after: {n_empty_after}")
 
 
if __name__ == "__main__":
    main()

