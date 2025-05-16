import requests

# Datos de prueba para Modelo 2
payload = {
    "local":            "Real Madrid",
    "visitor":          "Barcelona",
    "date":             "2025-05-10",
    "local_players":    [  # nombres normalizados (lowercase & strip)
        "vinícius jr", "rodrygo", "modrić", "kroos", "bale",
        "ramos", "karim benzema", "casemiro", "marcelo", "courtouis", "carvajal"
    ],
    "visitor_players":  [],  # puedes poner lista vacía si no los tienes
    "referee":          "mateu lahoz",
    "competition":      "first division",
    "cuota_1":          1.8,  # opcional: cuota casa
    "cuota_x":          3.5,  # opcional: cuota empate
    "cuota_2":          4.2   # opcional: cuota visitante
}

resp = requests.post(
    "http://127.0.0.1:8002/match_predict",  # puerto de Modelo 2
    json=payload
)

print(resp.status_code, resp.json())
