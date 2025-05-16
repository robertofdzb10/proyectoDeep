# test_agent.py

import requests
import numpy as np

AGENT_URL = "http://localhost:8002/predict"

def test_model1_low_level():
    """1) Low-level Modelo 1 → /predict"""
    payload = {
        "seq_loc": np.random.rand(10, 2).tolist(),
        "seq_vis": np.random.rand(10, 2).tolist(),
        "idx_loc": 3,
        "idx_vis": 7
    }
    r = requests.post(AGENT_URL, json=payload)
    print("1) Modelo 1 LOW-LEVEL:", r.status_code, r.json())

def test_model1_high_level():
    """2) High-level Modelo 1 → /match_predict"""
    payload = {
        "local":   "Real Madrid",
        "visitor": "Barcelona",
        "date":    "2025-05-10"
    }
    r = requests.post(AGENT_URL, json=payload)
    print("2) Modelo 1 HIGH-LEVEL:", r.status_code, r.json())

def test_model2_high_level():
    """3) High-level Modelo 2 → /match_predict"""
    payload = {
        "local":           "Real Madrid",
        "visitor":         "Barcelona",
        "date":            "2025-05-10",
        "local_players":   ["vini","rodrygo","modric"],
        "visitor_players": ["lewandowski","pedri","gavi"],
        "referee":         "Mateu Lahoz",
        "competition":     "La Liga",
        "cuota_1":         2.5,
        "cuota_x":         3.0,
        "cuota_2":         2.8
    }
    r = requests.post(AGENT_URL, json=payload)
    print("3) Modelo 2 HIGH-LEVEL:", r.status_code, r.json())

def test_model2_low_level():
    """4) Low-level Modelo 2 → /predict"""
    payload = {
        "seq_loc":    np.random.rand(15, 5).tolist(),
        "seq_vis":    np.random.rand(15, 5).tolist(),
        "idx_loc":    3,
        "idx_vis":    7,
        "idx_ref":    0,
        "idx_cmp":    1,
        "lineup_loc": list(range(11)),
        "lineup_vis": list(range(11,22)),
        "odds":       [0.30,0.35,0.35]
    }
    r = requests.post(AGENT_URL, json=payload)
    print("4) Modelo 2 LOW-LEVEL:", r.status_code, r.json())

if __name__ == "__main__":
    test_model1_low_level()
    test_model1_high_level()
    test_model2_high_level()
    test_model2_low_level()
