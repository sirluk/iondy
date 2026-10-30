# Sources for this post

The data and scripts behind the post's numbers and figures. The site builds only `../index.md` and the images it shows; nothing here is published as a page.

- `data/memory_matched_per_problem.csv`: every student's score on every problem, with answer lengths in tokens.
- `data/memory_matched_math_completions.csv`: every math answer, 16 per problem: its length, whether it is correct and answered, and how repetitive its last 4,000 characters are.
- `data/speed_all_layers.csv`: prefill and decode timings of SWA and Bifocal, medians of five requests at each prompt length.
- `data/post_numbers.json`: every number the post cites, written by `analysis/post_numbers.py`.
- `make_figures.py`: draws the nine figures into `../images`. The GDN + attention and full-attention speeds are written out in it.
- `analysis/load_samples.py`, `analysis/math_rollouts.py` and `analysis/write_tables.py`: how the two `memory_matched_*` tables were made from the evaluation output bundles, which are not included.

## Rebuild

With Python 3.10 or newer, numpy and matplotlib, from this folder:

```sh
python -I analysis/post_numbers.py
python -I make_figures.py
```

Both reproduce the committed files exactly on macOS. The figures use Helvetica Neue and Arial; where those are missing, matplotlib falls back to DejaVu Sans and the images come out slightly different.
