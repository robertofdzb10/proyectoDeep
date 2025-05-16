# test_agent.py

import requests
import numpy as np

AGENT_URL = "http://localhost:8002/predict"

def test_model1():
    """1) Modelo 1 → /match_predict"""
    payload = {
        "local":   "Real Madrid",
        "visitor": "Barcelona",
        "date":    "2025-05-10"
    }
    r = requests.post(AGENT_URL, json=payload)
    print("1) Modelo 1:", r.status_code, r.json())

def test_model2():
    """2) Modelo 2 → /match_predict"""
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
    print("2) Modelo 2:", r.status_code, r.json())

if __name__ == "__main__":
    test_model1()
    test_model2()
