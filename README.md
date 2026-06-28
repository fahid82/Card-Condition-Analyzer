# AI Trading Card Condition Analyzer (MVP)
This project is a simple FastAPI web app that estimates a trading card grading range from an uploaded image.

## What it does
Pipeline:
1. User uploads a card image.
2. OpenCV preprocessing:
   - Resize image.
   - Detect edges.
   - Detect/crop card from background (perspective correction when possible).
3. Feature extraction:
   - Centering estimate (`left/right`, `top/bottom` ratios).
   - Edge + corner wear heuristics.
4. Surface analysis:
   - AI-based surface scoring using Hugging Face Inference zero-shot labels (if configured).
   - Automatic fallback heuristic analysis if no API token is set.
5. Rule-based grade estimate:
   - Centering determines base score.
   - Edge/corner wear can cap score.
   - Surface damage subtracts points.
   - Final result returned as grade range (example: `8-9`).

## Tech stack
- FastAPI backend
- OpenCV + NumPy image processing
- Hugging Face Inference API (optional, free-tier capable with account/token)
- Basic HTML/CSS/vanilla JS frontend

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: enable AI surface API
```bash
cp .env.example .env
# then edit .env and set HF_TOKEN
```

## Run
```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## API
- `POST /analyze` (multipart form field: `image`)
- Returns JSON with:
  - preprocessing metadata
  - centering + edge/corner features
  - surface analysis (AI or heuristic source)
  - grade score/range and reasons

## Notes and limitations
- This is an MVP with deterministic rules and heuristic CV signals.
- Results are estimates and not a replacement for professional grading workflows.
- Centering detection assumes visible border transitions and can be less reliable on uncommon card layouts/backgrounds.
