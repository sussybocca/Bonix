from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os, requests, socket, subprocess

from dotenv import load_dotenv
load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set! Please add it to .env (local) or Railway secrets.")

# --------------------------------------------------------------------
# AI Models info
# --------------------------------------------------------------------
MODEL_INFO = {
    # Old chat models
    "meta-llama/Llama-3.1-8B-Instruct": {"api_key": "Llama3#rx5$tkadDl45%", "type": "chat", "is_space": False},
    "deepseek-ai/DeepSeek-V3-0324": {"api_key": "DeepSeek#rx5$tkadDl45%", "type": "chat", "is_space": False},
    "cognitivecomputations/dolphin-2.9.1": {"api_key": "Dolphin#rx5$tkadDl45%", "type": "chat", "is_space": False},
    # New text models
    "vngrs-ai/Kumru-2B": {"api_key": "Kumru#rx5$tkadDl45%", "type": "text", "is_space": False},
    "opendatalab/MinerU2.5-2509-1.2B": {"api_key": "Miner#rx5$tkadDl45%", "type": "text", "is_space": False},
    "deepseek-ai/DeepSeek-R1": {"api_key": "DeepSeekR1#rx5$tkadDl45%", "type": "text", "is_space": False},
}

# --------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------
app = FastAPI(title="Bonix API - Inference Providers")

@app.get("/")
async def root():
    return FileResponse("index.html")

# --------------------------------------------------------------------
# Request model for AI inference
# --------------------------------------------------------------------
class RunModelRequest(BaseModel):
    model: str
    input: str
    api_key: str

# --------------------------------------------------------------------
# Run AI model
# --------------------------------------------------------------------
@app.post("/api/run_model")
async def run_model(req: RunModelRequest):
    if req.model not in MODEL_INFO:
        raise HTTPException(status_code=404, detail="Model not found")

    if req.api_key != MODEL_INFO[req.model]["api_key"]:
        raise HTTPException(status_code=401, detail="Invalid API key")

    model_data = MODEL_INFO[req.model]
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {}

    if model_data["type"] == "chat" and not model_data["is_space"]:
        payload = {"model": req.model, "messages": [{"role": "user", "content": req.input}], "stream": False}
        endpoint = HF_CHAT_URL
    elif model_data["is_space"]:
        payload = {"inputs": req.input}
        endpoint = f"https://hf.space/embed/{req.model}/api/predict/"
    else:
        payload = {"inputs": req.input}
        endpoint = f"https://api-inference.huggingface.co/models/{req.model}"

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if model_data["type"] == "chat" and not model_data["is_space"]:
            output = result["choices"][0]["message"]["content"]
        elif model_data["is_space"]:
            output = result.get("data", result)
        else:
            output = result

        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

# --------------------------------------------------------------------
# List AI models
# --------------------------------------------------------------------
@app.get("/api/list_models")
async def list_models():
    return [
        {"model": m, "api_key": MODEL_INFO[m]["api_key"], "type": MODEL_INFO[m]["type"], "is_space": MODEL_INFO[m]["is_space"]}
        for m in MODEL_INFO.keys()
    ]

# --------------------------------------------------------------------
# Live Flask apps management
# --------------------------------------------------------------------
flask_apps = {}  # {app_id: {"process": Popen, "port": int}}

def get_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

@app.post("/api/run_flask_app")
async def run_flask_app(code: str = Form(...)):
    app_id = str(len(flask_apps) + 1)
    filename = f"user_app_{app_id}.py"

    with open(filename, "w") as f:
        f.write(code)

    port = get_free_port()
    proc = subprocess.Popen(["python", filename], env={**os.environ, "FLASK_RUN_PORT": str(port)})
    flask_apps[app_id] = {"process": proc, "port": port}
    url = f"https://bonix-production.up.railway.app/flask/{app_id}/"
    return JSONResponse({"app_id": app_id, "url": url})

@app.api_route("/flask/{app_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_flask(app_id: str, path: str, request: Request):
    if app_id not in flask_apps:
        raise HTTPException(status_code=404, detail="App not found")
    port = flask_apps[app_id]["port"]
    url = f"http://127.0.0.1:{port}/{path}"

    import httpx
    async with httpx.AsyncClient() as client:
        req_method = request.method.lower()
        req_func = getattr(client, req_method)
        response = await req_func(url, content=await request.body(), headers=request.headers)
        return JSONResponse(content=response.text, status_code=response.status_code)

# --------------------------------------------------------------------
# Run locally
# --------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))