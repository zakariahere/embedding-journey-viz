# The WHOLE WHOLE Journey - visual walkthrough

A tiny, zero-dependency web app that mirrors
step07c_the_whole_whole_journey.py (github.com/zakariahere/tokenization-llm)
visually, one step at a time, with animations and live heatmaps:

    text -> token IDs -> sliding windows -> a batch
         -> token embeddings (+256 numbers of MEANING per token)
         -> + positional embeddings (WHERE each token sits)
         -> [batch_size, max_length, 256]  <- what attention consumes next

## Run it

    python app.py

then open http://127.0.0.1:8765 in your browser.

Needs Python 3.9+ with torch and tiktoken installed (the same libraries the
reference script uses). Everything else is Python's standard library - no Flask,
no npm, no build step.

## What you see

The app walks the exact same acts as the script, each with its own view:

| Step | View |
|------|------|
| 1 - text | the-verdict.txt as plain text, character count |
| 2 - tokenize | the first 220 tokens as colored chips (text on top, ID below) |
| 3 - windows | a sliding window animating over the token strip - blue box = inputs, dashed amber box = targets (shifted +1) |
| 4+5 - batch | the [8, 4] batch grid, row 0's 4 next-token questions |
| 6 - token embeddings | ID to 256-vector lookup, and the whole batch as a heatmap wall (64 of 256 dims, chunk-averaged, for display) |
| 7 - positional | the 4 seat stickers, the token + seat = input animation, and the final input-embedding wall |
| proof | " like" in all 4 slots: identical vectors until position is added |
| recap | the whole pipeline on one screen -> [8, 4, 256] |

Header badges keep the five numbers that cause all the confusion visible at all
times: 1286 (windows, from the file), 8 (batch_size, your knob),
4 (max_length), 256 (embedding dim), [8, 4, 256] (the tensor).


## Paste your own text

Open "your own text" in the bottom bar, paste anything you like and hit
"run journey with my text" - the whole pipeline recomputes on your words:
real GPT-2 token IDs, windows, batch, embeddings, the works. The text is sent
as a POST body, so even long texts work. Clear the box and re-run to go back
to the-verdict.txt. Too-short text gets a friendly error instead of a crash.

## Tune the knobs

Open "tune the knobs" to change batch_size (1-16) and max_length (2-12) and
re-run the whole journey live - every step, shape and heatmap recomputes. Same
seed (123) as the reference script, so the numbers match it exactly.

## Files

- app.py - stdlib HTTP server + the journey computation (same logic and seed as the reference script)
- static/index.html - the whole frontend (no frameworks)
- the-verdict.txt - the input story (from the same repo)
- step07c_reference.py - the original console script, for comparison

## Controls

- Prev / Next buttons or the arrow keys step through the journey
- Play auto-advances through all 8 views
- Click any stage chip in the header to jump
- Deep-link any step with a URL hash, e.g. http://127.0.0.1:8765/#4