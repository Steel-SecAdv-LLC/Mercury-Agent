# Review samples

Three small files to try `mercury-agent review` (or the console's **REVIEW A
FILE**, or `POST /api/v1/review/upload`) on, one per kind:

| File | What it is | Planted in it (`planted.json`) |
|---|---|---|
| `sample_document.txt` | a run of prose from this repository's documentation | a sentence repeated over and over (characters 1705-3621) and a passage with its letters scrambled (5313-6476) |
| `sample_image.png` | a generated texture, 501 x 320, grayscale | a patch of a different texture (x 45-192, y 117-171) |
| `sample_recording.wav` | a generated voiced signal, 15.6 s, 16 kHz mono | a stretch of another voice (8.22-9.54 s) and an overdriven stretch (13.24-13.93 s) |

```bash
mercury-agent review examples/data/review/sample_document.txt
mercury-agent review examples/data/review/sample_image.png
mercury-agent review examples/data/review/sample_recording.wav
```

Each is a *spliced* reference artifact, so a review can be checked against
what was put in. Reviewed with the shipped checkpoints on 2026-11-21, all
three are grounded decisions (`act`) whose flagged segments lie inside the
planted spans: 10 of 48 document segments, 4 of 104 image tiles, 6 of 47
audio blocks.

They are examples, chosen to be small and to show a grounded decision; they
are not a measurement. How often a planted part is found, missed or abstained
on is in `artifacts/artifact_review.json`
(`python -m research.review.harness_artifact_review`).

Regenerate them (same text, pixels and samples; a PNG's compressed bytes can
differ between Pillow versions) with:

```bash
python -m research.review.reference_artifacts --write-samples examples/data/review
```
