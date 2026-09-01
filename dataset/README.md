# Local Data Layout

This directory is a local mount point for challenge controls and permitted public support data. Data files are intentionally excluded from Git because they are large and may have redistribution restrictions.

See `docs/REPRODUCIBILITY.md` for the expected layout and acquisition workflow. Never commit raw data, derived single-cell matrices, or credentials here.

The V7 scDFM research lane uses the following ignored local layout:

```text
dataset/
  external_repos/scDFM/        pinned official source checkout
  scdfm/downloads/             original immutable ZIP archives
  scdfm/extracted/             authenticated extracted data and weights
  scdfm/receipts/              local source, archive, and compatibility receipts
```

Run the tracked authenticators before using any of these files. The upstream
source-code license does not by itself establish a license for the downloaded
datasets or model weights, so redistribution remains disabled pending a
separate provenance review.
