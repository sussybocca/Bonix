from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os, requests, uuid
from dotenv import load_dotenv
import subprocess

# --------------------------------------------------------------------
# Load environment variables
# --------------------------------------------------------------------
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set! Please add it to .env or Railway secrets.")

# --------------------------------------------------------------------
# Models info: key + type + is_space
# --------------------------------------------------------------------
MODEL_INFO = {
    "meta-llama/Llama-3.1-8B-Instruct": {"api_key": "Llama3#rx5$tkadDl45%", "type": "chat", "is_space": False},
    "deepseek-ai/DeepSeek-V3-0324": {"api_key": "DeepSeek#rx5$tkadDl45%", "type": "chat", "is_space": False},
    "cognitivecomputations/dolphin-2.9.1": {"api_key": "Dolphin#rx5$tkadDl45%", "type": "chat", "is_space": False},
    "vngrs-ai/Kumru-2B": {"api_key": "Kumru#rx5$tkadDl45%", "type": "text", "is_space": False},
    "opendatalab/MinerU2.5-2509-1.2B": {"api_key": "Miner#rx5$tkadDl45%", "type": "text", "is_space": False},
    "deepseek-ai/DeepSeek-R1": {"api_key": "DeepSeekR1#rx5$tkadDl45%", "type": "text", "is_space": False},
}

# --------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------
app = FastAPI(title="Bonix AI Hub")

@app.get("/")
async def root():
    return FileResponse("index.html")

# --------------------------------------------------------------------
# AI Model Request
# --------------------------------------------------------------------
class RunModelRequest(BaseModel):
    model: str
    input: str
    api_key: str

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

    model_data = MODEL_INFO[req.model]
    model_type = model_data["type"]
    is_space = model_data["is_space"]

    if model_type == "chat" and not is_space:
        payload = {"model": req.model, "messages": [{"role": "user", "content": req.input}], "stream": False}
        endpoint = HF_CHAT_URL
    else:
        payload = {"inputs": req.input}
        endpoint = f"https://api-inference.huggingface.co/models/{req.model}"

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        output = result["choices"][0]["message"]["content"] if model_type == "chat" and not is_space else result
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

@app.get("/api/list_models")
async def list_models():
    return [
        {"model": m, "api_key": MODEL_INFO[m]["api_key"], "type": MODEL_INFO[m]["type"], "is_space": MODEL_INFO[m]["is_space"]}
        for m in MODEL_INFO.keys()
    ]

# --------------------------------------------------------------------
# New: Run Flask App
# --------------------------------------------------------------------
@app.post("/api/run_flask_app")
async def run_flask_app(code: str = Form(...)):
    """
    Takes Python Flask code from user, writes it to a file, runs it in a subprocess,
    and returns a permanent URL (simulated for Railway or hosting environment).
    """
    try:
        unique_id = str(uuid.uuid4())[:8]
        app_dir = f"flask_apps/{unique_id}"
        os.makedirs(app_dir, exist_ok=True)
        file_path = os.path.join(app_dir, "app.py")

        with open(file_path, "w") as f:
            f.write(code)

        # Run Flask app in a subprocess (production: would use proper WSGI hosting)
        subprocess.Popen(["python", file_path], cwd=app_dir)

        # Return simulated permanent URL
        flask_url = f"https://your-app-hosting/{unique_id}"
        return JSONResponse({"flask_url": flask_url})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flask app failed: {e}")

# --------------------------------------------------------------------
# Run locally
# --------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
