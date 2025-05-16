# api/router_api.py

from fastapi import FastAPI, HTTPException, Request
from api.schemas import MatchRequest
import httpx

app = FastAPI(title="Football Prediction Router API")

# URLs donde corren tus APIs de modelo
MODEL1_URL = "http://localhost:8000"    # Modelo 1 – low/high level
MODEL2_URL = "http://localhost:8001"    # Modelo 2 – low/high level

@app.get("/")
def root():
    return {"status": "ok", "router": "Endpoints for Model 1 & Model 2"}

@app.post("/predict")
async def route_predict(request: Request):
    payload = await request.json()
    keys = set(payload.keys())

    # 1) High-level Model 1: sólo local+visitor+date
    if keys.issubset({"local", "visitor", "date", "referee", "competition"}):
        print("🔄 Routing to Model 1 HIGH-LEVEL /match_predict")
        try:
            req = MatchRequest(**payload)
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{MODEL1_URL}/match_predict", json=req.dict())
            resp.raise_for_status()
            data = resp.json()
            data["routed_to"] = "model1_high"
            return data
        except Exception as e:
            raise HTTPException(400, f"Model1 high-level error: {e}")

    # 2) High-level Model 2: trae alineaciones y/o cuotas
    try:
        req2 = MatchRequest(**payload)
        print("🔄 Routing to Model 2 HIGH-LEVEL /match_predict")
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{MODEL2_URL}/match_predict", json=req2.dict())
        resp.raise_for_status()
        data = resp.json()
        data["routed_to"] = "model2_high"
        return data
    except Exception:
        ...

    print("❌ No model matched payload schema")
    raise HTTPException(400, "Formato de petición no válido para ningún modelo")
