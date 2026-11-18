# research-log

Working notes for this project are kept locally and are **not** published in this repository.

This file records that decision and the line it draws, so that a later reader does not mistake the missing history for an oversight.

## What is published

Only material judged necessary to reproduce the research:

- the code needed to run the pipeline, copied from the local working tree without edits
- the tests that exercise it
- design notes and guides that explain why the pipeline is shaped the way it is, without the chronology of arriving there
- the one reviewed demo clip in `docs/media/`, published with the author's explicit approval

## What is kept local

The local working tree holds the full record: dated failure reports, experiment logs, budget ledgers, superseded plans, and intermediate attempt notes. Those are the process of getting here, not the result. Publishing them wholesale would make the repository a development diary instead of a body of work, and would make the reproducible parts harder to find.

Private paths, absolute home directories, and per-session artifact directories are never published. Where a path is needed, the code resolves it through `H3_PRIVATE_ROOT` (see [`research/h3_personal/private_root.py`](../research/h3_personal/private_root.py)) and the documents refer to that variable instead of a literal location.

## Rule for what goes here

If a note is only useful for understanding how a decision was reached, it stays local. If it is needed to run, verify, or reproduce the result, it belongs in `docs/`. When the answer is unclear, the material stays local.
