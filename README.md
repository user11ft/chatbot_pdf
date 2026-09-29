# 📄 PDF Chatbot (Gradio + Gemini + FAISS)

Upload any PDF and chat with it. Answers include the source page numbers.

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then edit .env and add your GOOGLE_API_KEY
python app.py
```

Open http://127.0.0.1:7860

The first run downloads the embedding model (~90 MB), so it takes a moment.

## Deploy on Hugging Face Spaces (free)

1. Create a new Space -> SDK: **Gradio**.
2. Upload `app.py` and `requirements.txt`.
3. Space **Settings -> Variables and secrets -> New secret**:
   name `GOOGLE_API_KEY`, value = your key.
4. The Space builds and starts automatically.

## How it works

1. **Upload** -> PDF is read page by page (empty pages dropped) and split into chunks.
2. Chunks are embedded (`all-MiniLM-L6-v2`) and stored in a FAISS index — one per user session.
3. **Ask** -> the 4 most relevant chunks are retrieved and sent to Gemini together with
   the recent conversation, so follow-ups like "give another example" work.
4. The answer streams back with the page numbers it came from.

## Limitations

- Scanned PDFs (images only) have no extractable text and need OCR first.
- The index lives in memory: it's lost when the app restarts or the session ends.
