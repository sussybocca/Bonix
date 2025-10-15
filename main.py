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
# Models info: key + type
# --------------------------------------------------------------------
MODEL_INFO = {
    # Old chat models
    "meta-llama/Llama-3.1-8B-Instruct": {"api_key": "Llama3#rx5$tkadDl45%", "type": "chat"},
    "deepseek-ai/DeepSeek-V3-0324": {"api_key": "DeepSeek#rx5$tkadDl45%", "type": "chat"},
    "cognitivecomputations/dolphin-2.9.1": {"api_key": "Dolphin#rx5$tkadDl45%", "type": "chat"},

    # New Beta 1.2 models
    "emoji-gemma": {"api_key": "Emoji#rx5$tkadDl45%", "type": "text"},  # plain text input
    "arena": {"api_key": "Arena#rx5$tkadDl45%", "type": "text"},
    "Fathom-Search-4B": {"api_key": "Fathom#rx5$tkadDl45%", "type": "text"},
    "Ziya-Coding-34B": {"api_key": "Ziya#rx5$tkadDl45%", "type": "text"},
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
# Run model
# --------------------------------------------------------------------
@app.post("/api/run_model")
async def run_model(req: RunModelRequest):
    if req.model not in MODEL_INFO:
        raise HTTPException(status_code=404, detail="Model not found")

    if req.api_key != MODEL_INFO[req.model]["api_key"]:
        raise HTTPException(status_code=401, detail="Invalid API key")

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    model_type = MODEL_INFO[req.model]["type"]

    if model_type == "chat":
        # Chat LLMs
        payload = {
            "model": req.model,
            "messages": [{"role": "user", "content": req.input}],
            "stream": False
        }
        endpoint = HF_CHAT_URL
    else:
        # Non-chat models (text, search, code, etc.)
        payload = {"inputs": req.input}
        endpoint = f"https://api-inference.huggingface.co/models/{req.model}"

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if model_type == "chat":
            output = result["choices"][0]["message"]["content"]
        else:
            # For non-chat models, just return the JSON result directly
            output = result

        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

# --------------------------------------------------------------------
# List models
# --------------------------------------------------------------------
@app.get("/api/list_models")
async def list_models():
    return [{"model": m, "api_key": MODEL_INFO[m]["api_key"], "type": MODEL_INFO[m]["type"]} for m in MODEL_INFO.keys()]

# --------------------------------------------------------------------
# Run locally
# --------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
