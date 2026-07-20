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

Data in the Normalized folder obtained by running: `./run_normalize.sh ./Processed ./Normalized`