from fastapi import FastAPI, HTTPException 
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os, requests
from dotenv import load_dotenv

# --------------------------------------------------------------------
# Load environment variables
# --------------------------------------------------------------------
# Only loads .env locally; on Railway, environment variables are injected automatically
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set! Please add it to .env (local) or Railway secrets.")

# --------------------------------------------------------------------
# Models and API keys
# --------------------------------------------------------------------
MODEL_KEYS = {
    # Old models
    "meta-llama/Llama-3.1-8B-Instruct": "Llama3#rx5$tkadDl45%",
    "deepseek-ai/DeepSeek-V3-0324": "DeepSeek#rx5$tkadDl45%",
    "cognitivecomputations/dolphin-2.9.1": "Dolphin#rx5$tkadDl45%",
    
    # New models for Beta 1.2
    "emoji-gemma": "Emoji#rx5$tkadDl45%",
    "arena": "Arena#rx5$tkadDl45%",
    "Fathom-Search-4B": "Fathom#rx5$tkadDl45%",
    "Ziya-Coding-34B": "Ziya#rx5$tkadDl45%",
}

# --------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------
app = FastAPI(title="Bonix API - Inference Providers")

@app.get("/")
async def root():
    return FileResponse("index.html")

# --------------------------------------------------------------------
# Request model
# --------------------------------------------------------------------
class RunModelRequest(BaseModel):
    model: str
    input: str
    api_key: str

# --------------------------------------------------------------------
# Run model (via Hugging Face Router)
# --------------------------------------------------------------------
@app.post("/api/run_model")
async def run_model(req: RunModelRequest):
    if req.model not in MODEL_KEYS:
        raise HTTPException(status_code=404, detail="Model not found")

    if req.api_key != MODEL_KEYS[req.model]:
        raise HTTPException(status_code=401, detail="Invalid API key")

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": req.model,
        "messages": [
            {"role": "user", "content": req.input}
        ],
        "stream": False
    }

    try:
        response = requests.post(HF_CHAT_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        output = result["choices"][0]["message"]["content"]
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

# --------------------------------------------------------------------
# List models
# --------------------------------------------------------------------
@app.get("/api/list_models")
async def list_models():
    return [{"model": m, "api_key": MODEL_KEYS[m]} for m in MODEL_KEYS.keys()]

# --------------------------------------------------------------------
# Run locally
# --------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
