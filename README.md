# proyectoDeep

Un sistema de predicción de resultados de fútbol basado en dos modelos de Deep Learning, un router/agent FastAPI y un frontend de chat minimalista.

---

## 🚀 Características

- **Modelo 1**: API REST Low-Level y High-Level para predicciones basadas solo en equipos y fecha.  
- **Modelo 2**: API REST Low-Level y High-Level que además incorpora alineaciones, árbitro, competición y cuotas.  
- **Router API**: un router FastAPI que redirige cada petición al endpoint adecuado (modelo 1 o 2).  
- **LangChain Agent**: un agente conversacional que extrae entidades (equipos, jugadores, cuotas…) y llama al Router para obtener la predicción.  
- **Frontend**: chat web standalone en `frontend/index.html` para interactuar con el LangChain Agent.  
- **Scripts de arranque**: `run_all.ps1` (Windows PowerShell) lanza los cuatro servicios en background.

---

## 📁 Estructura

```

proyectoDeep/
├─ api/
│  ├─ model1\_api.py         # FastAPI Modelo 1
│  ├─ model2\_api.py         # FastAPI Modelo 2
│  ├─ router\_api.py         # FastAPI Router entre modelo 1 y 2
│  └─ langchain\_agent.py    # FastAPI LangChain Agent
├─ agents/
│  └─ agent.py              # Código del agente LangChain
├─ core/
│  ├─ data\_processing\_model1.py
│  ├─ inference\_model1.py
│  ├─ data\_processing\_model2.py
│  └─ inference\_model2.py
├─ frontend/
│  └─ index.html            # Chat minimalista HTML+JS
├─ test/
│  ├─ test1.py              # pruebas unitarias / smoke tests
│  └─ test\_agent.py         # pruebas LangChain Agent
├─ utils/
│  └─ generate\_model2\_artifacts.py
├─ model/                   # pesos y artefactos (artefacts.pkl, .pth, configs…)
├─ data/                    # CSVs, parquet, jugadores…
├─ run\_all.ps1              # PowerShell launcher (4 ventanas minimizadas)
├─ requirements.txt
└─ README.md                # ← este archivo

````

---

## 🛠️ Instalación y puesta en marcha

1. **Clona el repositorio**  
   ```bash
   git clone https://github.com/robertofdzb10/proyectoDeep.git
   cd proyectoDeep
````

2. **Crea y activa un entorno virtual (venv)**

   ```bash
   python -m venv venv
   . venv/Scripts/activate    # Windows PowerShell
   source venv/bin/activate   # macOS / Linux
   ```

3. **Instala dependencias**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Genera artefactos del Modelo 2** *(solo si no tienes `model/artefacts.pkl` y `history_df.parquet`)*

   ```bash
   python utils/generate_model2_artifacts.py
   ```

5. **Lanza todos los servicios**

   * **Windows**:

     ```powershell
     .\run_all.ps1
     ```

     Esto levantará en segundo plano:

     * Modelo1 API → [http://localhost:8000](http://localhost:8000)
     * Modelo2 API → [http://localhost:8001](http://localhost:8001)
     * Router API  → [http://localhost:8002](http://localhost:8002)
     * LangChain Agent API → [http://localhost:8003](http://localhost:8003)
   * **macOS/Linux** (alternativa manual):

     ```bash
     uvicorn api.model1_api:app --port 8000 --reload &
     uvicorn api.model2_api:app --port 8001 --reload &
     uvicorn api.router_api:app --port 8002 --reload &
     uvicorn api.langchain_agent:app --port 8003 --reload &
     ```

---

## 💬 Uso

1. **Interfaz Web**
   Abre `frontend/index.html` en el navegador. Empieza a chatear de inmediato: el frontend hace POST a `http://localhost:8003/predict`.

2. **Endpoints REST (para integración)**

   * **Modelo 1 High-Level**:
     `POST http://localhost:8000/match_predict`
   * **Modelo 2 High-Level**:
     `POST http://localhost:8001/match_predict`
   * **Router/Agent**:
     `POST http://localhost:8002/predict` recibe un JSON genérico y reenvía al modelo apropiado.
   * **LangChain Agent**:
     `POST http://localhost:8003/predict` con

     ```json
     { "input": "Real Madrid vs Barcelona el 2025-05-10 con alineación …" }
     ```

     Devuelve texto conversacional.

---

## 🧪 Tests

```bash
# Ejecuta los test de API & agente
python test/test1.py
python test/test2.py
python test/test3.py
python test/test_agent.py
```

---

## 🔧 Personalización

* Ajusta puertos o rutas en `run_all.ps1`.
* Modifica el prompt del agente en `agents/agent.py`.
* Cambia estilos del chat en `frontend/index.html`.
