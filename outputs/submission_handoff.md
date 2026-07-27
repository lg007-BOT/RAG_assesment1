# Submission handoff

## Complete these four steps

1. Add your selected 20+ page PDF or focused PDF set to `data/raw/`.
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
3. Run:

   ```bash
   source .venv/bin/activate
   python -m src.ingest
   python demo.py "<a question answered by the PDF>"
   python demo.py "Who won the 2022 FIFA World Cup?"
   ```

4. Replace the placeholder positive question in
   `tests/evaluation_questions.json` with document-specific questions and page
   numbers, then run `python evaluate.py`.

## What to submit

- Entire project folder as the code repository.
- `writeup.md` as the short design write-up.
- Loom video using `loom_script.md`.

## Verified locally

- The local unit test and Python syntax validation pass. Run `pip install -r
  requirements.txt` in an internet-enabled environment to install the declared
  Gemini SDK dependency.
- `python -m pytest -q` passes (1 test).
- Python syntax compilation passes.
- The project uses Gemini `generate_content` Structured Outputs with a JSON
  Schema derived from the Pydantic response model, then validates the returned
  JSON again in code.
