import requests

resp = requests.post(
    "http://localhost:8001/match_predict",
    json={"local":"Real Madrid","visitor":"Barcelona","date":"2025-05-10"}
)
print(resp.json())
