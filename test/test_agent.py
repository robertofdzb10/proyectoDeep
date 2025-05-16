# test_langchain_agent.py
import requests
import textwrap
import json

BASE_URL = "http://localhost:8003/predict"

def pretty(j):
    """Imprime JSON bonito en consola."""
    print(json.dumps(j, indent=2, ensure_ascii=False))

def test_high_level_model1():
    """Solo equipos + fecha → debería usar modelo 1."""
    prompt = (
        "¿Quién crees que ganará el partido entre el Real Madrid y el Barcelona "
        "el 2025-05-10?"
    )
    resp = requests.post(BASE_URL, json={"input": prompt})
    print("\n🟥  TEST 1  (High-level Modelo 1)")
    print("Prompt:", prompt)
    print("Status:", resp.status_code)
    pretty(resp.json())

def test_high_level_model2():
    """Con alineaciones/cuotas → debería usar modelo 2."""
    prompt = textwrap.dedent("""
        Analiza el partido del 2025-05-10 entre el Real Madrid y el Barcelona.
        Alineación local: [Vinícius Júnior, Rodrygo, Bellingham].
        Alineación visitante: [Lewandowski, Pedri, Gündogan].
        Árbitro: Mateu Lahoz.  Cuota_1 = 2.6, Cuota_X = 3.1, Cuota_2 = 2.75.
    """).strip()
    resp = requests.post(BASE_URL, json={"input": prompt})
    print("\n🟦  TEST 2  (High-level Modelo 2)")
    print("Prompt:", prompt)
    print("Status:", resp.status_code)
    pretty(resp.json())

def test_invalid_payload():
    """Caso sin equipos → agente debe pedir más datos o lanzar error."""
    prompt = "¿Cómo está el clima hoy?"
    resp = requests.post(BASE_URL, json={"input": prompt})
    print("\n🟧  TEST 3  (Payload inválido / sin equipos)")
    print("Prompt:", prompt)
    print("Status:", resp.status_code)
    pretty(resp.json())

if __name__ == "__main__":
    print("===== Comenzando tests LangChain Agent (puerto 8003) =====")
    test_high_level_model1()
    test_high_level_model2()
    test_invalid_payload()
    print("\n✅  Tests finalizados")
