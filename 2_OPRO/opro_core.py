import re 
import time 
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI 

import config as C 

EXTRACTION_STATS = Counter()

# Initial Rubric taken from https://aclanthology.org/N16-2013.pdf
INITIAL_RUBRIC_ENGLISH = """\
A text is hate speech if it:
1. uses a sexist or racial slur.
2. attacks a minority.
3. seeks to silence a minority.
4. criticizes a minority (without a well founded argument).
5. promotes, but does not directly use, hate speech or violent crime.
6. criticizes a minority and uses a straw man argument.
7. blatantly misrepresents truth or seeks to distort views on a minority with unfounded claims.
8. shows support of problematic hash tags. E.g. "#BanIslam", "#whoriental", "#whitegenocide".
9. negatively stereotypes a minority.
10. defends xenophobia or sexism.
11. contains a screen name that is offensive, as per the previous criteria, the tweet is ambiguous (at best), and the tweet is on a topic that satisfies any of the above criteria.\
"""

INITIAL_RUBRIC = """\
Suatu teks dikategorikan sebagai ujaran kebencian (hate speech) jika teks tersebut:
1. menggunakan kata makian/hinaan yang bersifat seksis atau rasis.
2. menyerang kelompok minoritas.
3. berupaya membungkam kelompok minoritas.
4. mengkritik kelompok minoritas (tanpa argumen yang berdasar).
5. mempromosikan ujaran kebencian atau kejahatan kekerasan, meskipun tidak menggunakannya secara langsung.
6. mengkritik kelompok minoritas dan menggunakan argumen 'straw man' (menguraikan/memutarbalikkan argumen lawan secara tidak jujur agar mudah diserang).
7. secara terang-terangan memutarbalikkan kebenaran atau berupaya mendistorsi pandangan terhadap kelompok minoritas dengan klaim yang tidak berdasar.
8. menunjukkan dukungan terhadap tagar-tagar bermasalah. Contoh: "#BanIslam", "#whoriental", "#whitegenocide".
9. memberikan stereotipe negatif terhadap kelompok minoritas.
10. membela sikap xenofobia (kebencian/ketakutan terhadap orang asing) atau seksisme.
11. mengandung nama pengguna (screen name) yang menyinggung sesuai kriteria sebelumnya, unggahan/tweet bersifat ambigu, dan unggahan tersebut membahas topik yang memenuhi salah satu kriteria di atas.\
"""

SCORER_TEMPLATE = """\
Your task is to classify whether the given text is considered hate speech or not. Use ONLY the rubric below to decide the label. Do not use any other knowledge of what hate speech is.
RUBRIC OF HATE SPEECH:
--- START OF RUBRIC ---
{HYPOTHESIS_RUBRIC}
--- END ---

TASK:
Read the given text. If it satisfies any criterion above, the label is "hate". Otherwise, the label is "not-hate".
 
You may reason about your decision first. Then give your final label wrapped in <answer> tags, containing ONLY the word hate or not-hate, like:
<answer>hate</answer>
or
<answer>not-hate</answer>

TEXT:
"{SCORER_INPUT}"\
"""

EXEMPLAR_TEMPLATE = """\
Example {N}:
Text: {EXEMPLAR_SAMPLE_TEXT}
Label: {EXEMPLAR_SAMPLE_LABEL}
"""
EXEMPLAR_TEMPLATE_SEPARATOR = "\n"

TRIED_RUBRICS_TEMPLATE = """\
{RUBRIC_STRING}
Score = {RUBRIC_SCORE}\
"""
TRIED_RUBRICS_TEMPLATE_SEPARATOR = "\n"

METAPROMPT_TEMPLATE = """\
You are refining a RUBRIC of criteria that define hate speech, to best match how a collection of social-media posts has been labelled.

The rubric is a numbered list of criteria. A post is labelled "hate" if it satisfies any criteria in the rubric, "non-hate" otherwise. A good rubric produces labels matching the collection's true labels.
You must INFER what this collection treats as hate from the examples and from which rubrics scored well. Criteria may need to be ADDED, REMOVED, REWORDED, MERGED, or SPLIT.

Scores are macro-F1 (0-100, higher is better). Rubrics are sorted worst (top) to best (bottom). The best so far is LAST.

--- TRIED RUBRICS (worst to best) ---
{TRIED_RUBRICS_STRING}
--- END ---

Here are example posts with their correct labels, to show what this collection treats as hate vs. non-hate:
--- START OF EXEMPLARS ---
{EXEMPLARS_STRING}
--- END ---

Write ONE NEW rubric that scores higher than the best above. It may add, cut, or revise criteria. At most {CRITERIA_CAP} criteria.

Provide the FINAL rubric wrapped in <rubric> tags. Inside the <rubric> tags, output ONLY a numbered list of criteria. Keep any reasoning OUTSIDE the tags. For example:
<rubric>
1. ...
2. ...
</rubric>\
"""

# Rubric Extraction
def strip_thinking_tags(text) -> str:
    # Remove <think>/<thought> blocks; tolerate non-string input (None/NaN).
    if not isinstance(text, str):
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        text = str(text)
    cleaned = re.sub(r"<(think|thought)>.*?</\1>", "", text, flags=re.DOTALL)
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[-1]
    if "<thought>" in cleaned:
        cleaned = cleaned.split("<thought>")[-1]
    return cleaned.strip()

def extract_tagged(text: str, tag: str):
    # Pull contents of the LAST <tag>...</tag> block; tolerate a missing closer.
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    if f"<{tag}>" in text:
        return text.split(f"<{tag}>")[-1].strip()
    return None

def _normalize_label_str(s: str):
    # Map to 'hate'/'not-hate' or None.
    t = s.casefold()
    if any(w in t for w in ["not-hate", "nothate", "not hate", "non-hate", "non hate"]):
        return C.NONHATE_LABEL
    if any(w in t for w in ["hate", "hateful"]):
        return C.HATE_LABEL
    return None

def parse_scorer_output(raw):
    cleaned = strip_thinking_tags(raw)
    if not cleaned:
        return None
    tagged = extract_tagged(cleaned, "answer")
    if tagged is not None:
        norm = _normalize_label_str(tagged)
        if norm is not None:
            EXTRACTION_STATS["scorer_tag_ok"] += 1
            return norm
        EXTRACTION_STATS["scorer_tag_unparseable"] += 1
    norm = _normalize_label_str(cleaned)
    if norm is not None:
        EXTRACTION_STATS["scorer_fallback_scan"] += 1
        return norm
    return None

def parse_optimizer_output(raw):
    cleaned = strip_thinking_tags(raw)
    if not cleaned:
        return None
    tagged = extract_tagged(cleaned, "rubric")
    if tagged:
        EXTRACTION_STATS["optimizer_tag_ok"] += 1
        return tagged
    numbered = [ln for ln in cleaned.splitlines()
                if re.match(r"\s*\d+\s*[\.\)\-:]", ln)]
    if numbered:
        EXTRACTION_STATS["optimizer_fallback_numbered"] += 1
        return "\n".join(numbered).strip()
    return None

# vLLM Client
class VLLMClient:
    def __init__(self, base_url: str = None, served_model_name: str = None):
        self.client = OpenAI(base_url=base_url or C.BASE_URL, api_key="EMPTY")
        self.model = served_model_name

    def call(self, prompt: str, seed: int, temperature: float,
             max_tokens: int):
        """One raw completion. content MAY be None (empty vLLM generation);
        callers handle None via the retry wrappers."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        return resp.choices[0].message.content

    # For retry purposes
    def score_label_retried(self, prompt: str, seed: int):
        """Scorer call + parse, retried on empty/unparseable output. Returns a
        canonical label or None (caller drops None rows). Deterministic (temp 0);
        the seed is nudged each retry so a re-issue isn't bit-identical."""
        for attempt in range(C.RETRY_MAX + 1):
            raw = self.call(prompt, seed=seed + attempt, temperature=0.0,
                            max_tokens=C.SCORER_MAX_TOKENS)
            label = parse_scorer_output(raw)
            if label is not None:
                if attempt > 0:
                    EXTRACTION_STATS["scorer_retry_succeeded"] += 1
                return label
            EXTRACTION_STATS["scorer_retry_attempt"] += 1
            time.sleep(C.RETRY_BACKOFF)
        EXTRACTION_STATS["scorer_dropped"] += 1
        return None

    def optimizer_rubric_retried(self, prompt: str, seed: int, temperature: float):
        """Optimizer call + parse, retried on empty/unparseable output. Returns a
        rubric string or None (caller skips that candidate)."""
        for attempt in range(C.RETRY_MAX + 1):
            raw = self.call(prompt, seed=seed + 1000 * attempt,
                            temperature=temperature, max_tokens=C.OPT_MAX_TOKENS)
            rubric = parse_optimizer_output(raw)
            if rubric is not None:
                if attempt > 0:
                    EXTRACTION_STATS["optimizer_retry_succeeded"] += 1
                return rubric
            EXTRACTION_STATS["optimizer_retry_attempt"] += 1
            time.sleep(C.RETRY_BACKOFF)
        EXTRACTION_STATS["optimizer_dropped"] += 1
        return None

def scorer_prompter(hypothesis_rubric: str) -> str:
    return SCORER_TEMPLATE.replace("{HYPOTHESIS_RUBRIC}", hypothesis_rubric)

def fill_scorer_input(scorer_prompt: str, text: str) -> str:
    return scorer_prompt.replace("{SCORER_INPUT}", text) 

def exemplar_stringify(exemplar_hate, exemplar_nonhate, n, seed, seed_step) -> str:
    """Sample n hate + n non-hate exemplars from the rung reservoir. Guarded so
    a low rung with fewer than n rows of a class samples with replacement rather
    than raising."""
    step_seed = seed + seed_step

    def _sample(pool):
        replace = len(pool) < n
        return pool.sample(n=n, replace=replace, random_state=step_seed)

    combined = pd.concat([_sample(exemplar_hate), _sample(exemplar_nonhate)],
                         ignore_index=True)
    blocks = []
    for i, row in enumerate(combined.itertuples(index=False), start=1):
        blocks.append(EXEMPLAR_TEMPLATE.format(
            N=i,
            EXEMPLAR_SAMPLE_TEXT=getattr(row, C.TEXT_COL),
            EXEMPLAR_SAMPLE_LABEL=getattr(row, C.LABEL_COL),
        ))
    return EXEMPLAR_TEMPLATE_SEPARATOR.join(blocks)

def tried_rubrics_stringify(rubrics_dict, n) -> str:
    blocks = []
    for rank in range(n, 0, -1):
        entry = rubrics_dict[rank]
        blocks.append(TRIED_RUBRICS_TEMPLATE.format(
            RUBRIC_STRING=entry["rubric"], RUBRIC_SCORE=entry["score"]))
    return TRIED_RUBRICS_TEMPLATE_SEPARATOR.join(blocks)

def optimizer_prompter(rubrics_dict, exemplar_hate, exemplar_nonhate,
                       n_rubrics, n_exemplars, seed, seed_step, criteria_cap) -> str:
    tried = tried_rubrics_stringify(rubrics_dict, n_rubrics)
    exemplars = exemplar_stringify(
        exemplar_hate, exemplar_nonhate, n_exemplars, seed, seed_step)
    return METAPROMPT_TEMPLATE.format(
        TRIED_RUBRICS_STRING=tried, EXEMPLARS_STRING=exemplars,
        CRITERIA_CAP=criteria_cap)

def score_dataframe(df, client: VLLMClient, hypothesis_rubric: str,
                    seed: int, max_workers: int) -> pd.DataFrame:
    scorer_prompt = scorer_prompter(hypothesis_rubric)
    texts = [str(t) for t in df[C.TEXT_COL].tolist()]
    preds = [None] * len(texts)

    def score_one(idx_text):
        idx, text = idx_text
        full = fill_scorer_input(scorer_prompt, text)
        return idx, client.score_label_retried(full, seed=seed)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(score_one, (i, t)) for i, t in enumerate(texts)]
        for fut in as_completed(futures):
            idx, label = fut.result()
            preds[idx] = label

    out = df.copy()
    out["pred"] = preds
    return out
