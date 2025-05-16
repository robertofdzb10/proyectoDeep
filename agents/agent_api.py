# api/agent_api.py

from fastapi import FastAPI, HTTPException, Request
from api.schemas import (
    PredictRequestModel1,
    PredictRequestModel2,
    MatchRequest
)
import httpx

app = FastAPI(title="Model Agent API")

# URLs donde corren tus APIs de modelo
MODEL1_URL = "http://localhost:8000"    # Modelo 1
MODEL2_URL = "http://localhost:8001"    # Modelo 2

@app.get("/")
def root():
    return {"status": "ok", "agent": "Router between model1 and model2"}

@app.post("/predict")
async def agent_predict(request: Request):
    payload = await request.json()

    # 1) Low-level Modelo 1 → /predict
    try:
        body1 = PredictRequestModel1(**payload)
        print("🔄 Routing to Model 1 LOW-LEVEL /predict")
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{MODEL1_URL}/predict", json=body1.dict())
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        data["routed_to"] = "model1_low"
        return data
    except Exception:
        ...

    # 2) High-level Modelo 1 → /match_predict (solo local+visitor+date)
    keys = set(payload.keys())
    if keys.issubset({"local", "visitor", "date", "referee", "competition"}):
        try:
            match_req = MatchRequest(**payload)
            print("🔄 Routing to Model 1 HIGH-LEVEL /match_predict")
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{MODEL1_URL}/match_predict", json=match_req.dict())
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            data = resp.json()
            data["routed_to"] = "model1_high"
            return data
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 3) High-level Modelo 2 → /match_predict (con players/cuotas)
    try:
        match2 = MatchRequest(**payload)
        # si trae alineaciones o cuotas, cae aquí
        print("🔄 Routing to Model 2 HIGH-LEVEL /match_predict")
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{MODEL2_URL}/match_predict", json=match2.dict())
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        data["routed_to"] = "model2_high"
        return data
    except Exception:
        ...

    # 4) Low-level Modelo 2 → /predict
    try:
        body2 = PredictRequestModel2(**payload)
        print("🔄 Routing to Model 2 LOW-LEVEL /predict")
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{MODEL2_URL}/predict", json=body2.dict())
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        data["routed_to"] = "model2_low"
        return data
    except Exception:
        ...

    # Ningún esquema coincide
    print("❌ No model matched payload schema")
    raise HTTPException(400, "Formato de petición no válido para ningún modelo")
