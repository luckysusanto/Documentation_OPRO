# Normalization Checklist
- [ ] HTML unescape `&amp; -> & | &lt; -> < | etc`
- [ ] Unicode NFC (Decompose characters to one canonical form for tokenization)
- [ ] Strip retweet prerfix (Removes "RT" in `RT @USER`)
- [ ] URLs normalization (Normalize `http: | https: | www. | uniform resource locator` into "HTTPURL")
- [ ] Mentions (Convert mentioned users string into one "@USER")
- [ ] Hashtags (Removed `#`, but keep the text)
- [ ] Emojis (Convert to text form such as "loudly crying face", non-convertable symbols are removed entirely.)
- [ ] whitespace + edge-quote cleanup
- [ ] Casing (default to no casing, use --lowercase to lowercase the text. IndoBERTweet-uncased can automatically lowercase.)

Data in the Normalized folder obtained by running: `bash ./run_normalize.sh ./Processed ./Normalized`

---

# Token Statistics Breakdown

| Split / File | Rows | Mean Tokens | P95 | P99 | Max | $\le$ 512 (%) | > 512 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`csis_exemplars_hate_100`** | 6,698 | 24.6 | 54 | 63 | 85 | **100.0%** | 0.0% |
| **`csis_exemplars_nonhate_100`** | 6,698 | 28.0 | 56 | 66 | 135 | **100.0%** | 0.0% |
| **`csis_test_set`** | 500 | 26.8 | 57 | 65 | 76 | **100.0%** | 0.0% |
| **`csis_val_set`** | 250 | 26.3 | 56 | 61 | 75 | **100.0%** | 0.0% |
| **`indoDiscourse_exemplars_hate_100`** | 3,495 | 59.8 | 199 | 399 | 1,014 | **99.5%** | 0.5% (19) |
| **`indoDiscourse_exemplars_nonhate_100`** | 3,495 | 78.1 | 290 | 685 | 1,650 | **98.1%** | 1.9% (65) |
| **`indoDiscourse_test_set`** | 500 | 69.1 | 220 | 450 | 1,534 | **99.0%** | 1.0% (5) |
| **`indoDiscourse_val_set`** | 250 | 62.2 | 232 | 453 | 866 | **99.2%** | 0.8% (2) |

IndoBERTweet has a max_seq_length of 512. Since the amount of data points that are longer than this is minimal, just truncate them during feeding. **Report this as one of the weakness of transformer-based architectures.**