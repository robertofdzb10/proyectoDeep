# Importaciones necesarias
import json
import re
import requests
import random
from typing import Dict, List, Optional, Tuple, Any
from rapidfuzz import process, fuzz
from dotenv import load_dotenv
import os
from datetime import date

# Importaciones de LangChain
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import CallbackManager, StdOutCallbackHandler

# carga variables de .env
load_dotenv()  

ROUTER_URL = "http://localhost:8002/predict"   # ← endpoint del router

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY no encontrada en el entorno")

# Diccionario de equipos
TEAM_DICT = {
    "eibar": 0, "granada": 1, "cádiz": 2, "valencia": 3, "alavés": 4, "villarreal": 5,
    "valladolid": 6, "getafe": 7, "celta de vigo": 8, "betis": 9, "real sociedad": 10,
    "huesca": 11, "elche": 12, "osasuna": 13, "barcelona": 14, "atlético de madrid": 15,
    "real madrid": 16, "sevilla": 17, "athletic club": 18, "levante": 19, "mallorca": 20,
    "espanyol": 21, "rayo vallecano": 22, "almería": 23, "girona": 24, "las palmas": 25
}

# --------------------------
# Funciones de extracción de entidades
# --------------------------

def find_teams_in_text(user_input: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Encuentra los nombres de dos equipos en el texto del usuario.

    Args:
        user_input: Texto del mensaje del usuario

    Returns:
        Tupla con los nombres de los equipos (local, visitante) o (None, None) si no se encuentran
    """
    user_input = user_input.lower()
    teams = list(TEAM_DICT.keys())
    matches = process.extract(user_input, teams, scorer=fuzz.partial_ratio, limit=5)

    # Filtrar coincidencias con score alto
    matched = [team for team, score, _ in matches if score > 75]

    # Si encontramos al menos dos equipos, devolver los dos primeros
    if len(matched) >= 2:
        return matched[0], matched[1]

    return None, None

def extract_players_from_text(user_input: str) -> Tuple[List[str], List[str]]:
    """
    Extrae listas de jugadores del texto del usuario.
    """
    # Patrones para detectar listas de jugadores
    local_players = []
    visitor_players = []

    # Buscar patrones de listas de jugadores locales
    local_patterns = [
        r"jugadores?\s+locales?[:\s]+\[([^\]]+)\]",
        r"jugadores?\s+locales?[:\s]+(.+?)(?=jugadores?\s+visitantes?|$)",
        r"alineaci[óo]n\s+local[:\s]+\[([^\]]+)\]",
        r"alineaci[óo]n\s+local[:\s]+(.+?)(?=alineaci[óo]n\s+visitante|$)",
        r"equipo\s+local[:\s]+\[([^\]]+)\]",
        r"equipo\s+local[:\s]+(.+?)(?=equipo\s+visitante|$)",
        r"local_players\s*=\s*\[([^\]]+)\]"
    ]

    # Buscar patrones de listas de jugadores visitantes
    visitor_patterns = [
        r"jugadores?\s+visitantes?[:\s]+\[([^\]]+)\]",
        r"jugadores?\s+visitantes?[:\s]+(.+?)(?=$)",
        r"alineaci[óo]n\s+visitante[:\s]+\[([^\]]+)\]",
        r"alineaci[óo]n\s+visitante[:\s]+(.+?)(?=$)",
        r"equipo\s+visitante[:\s]+\[([^\]]+)\]",
        r"equipo\s+visitante[:\s]+(.+?)(?=$)",
        r"visitor_players\s*=\s*\[([^\]]+)\]"
    ]

    # Buscar jugadores locales
    for pattern in local_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            players_text = match.group(1)
            # Limpiar y dividir la lista de jugadores
            players = [p.strip().strip("'\"") for p in re.split(r',|\s+y\s+', players_text)]
            local_players = [p for p in players if p]  # Filtrar elementos vacíos
            break

    # Buscar jugadores visitantes
    for pattern in visitor_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            players_text = match.group(1)
            # Limpiar y dividir la lista de jugadores
            players = [p.strip().strip("'\"") for p in re.split(r',|\s+y\s+', players_text)]
            visitor_players = [p for p in players if p]  # Filtrar elementos vacíos
            break

    return local_players, visitor_players

def extract_all_entities(user_input: str) -> Dict[str, Any]:
    """
    Extrae todas las entidades relevantes del texto del usuario.
    """
    # Extraer equipos
    local_team, visitor_team = find_teams_in_text(user_input)

    # Extraer jugadores
    local_players, visitor_players = extract_players_from_text(user_input)

    # Extraer árbitro (simplificado para el ejemplo)
    referee_match = re.search(r"[áa]rbitro[:\s]+([^,\n.]+)", user_input, re.IGNORECASE)
    referee = referee_match.group(1).strip() if referee_match else None

    # Extraer competición (simplificado para el ejemplo)
    competition_match = re.search(r"competici[óo]n[:\s]+([^,\n.]+)", user_input, re.IGNORECASE)
    competition = competition_match.group(1).strip() if competition_match else "First Division"

    # Extraer fecha (simplificado para el ejemplo)
    date_match = re.search(r"fecha[:\s]+(\d{4}-\d{2}-\d{2})", user_input, re.IGNORECASE)
    match_date = date_match.group(1) if date_match else None

    # Extraer cuotas (simplificado para el ejemplo)
    cuota_1_match = re.search(r"cuota[_\s]*1\s*=\s*(\d+\.?\d*)", user_input, re.IGNORECASE)
    cuota_1 = float(cuota_1_match.group(1)) if cuota_1_match else None

    cuota_x_match = re.search(r"cuota[_\s]*x\s*=\s*(\d+\.?\d*)", user_input, re.IGNORECASE)
    cuota_x = float(cuota_x_match.group(1)) if cuota_x_match else None

    cuota_2_match = re.search(r"cuota[_\s]*2\s*=\s*(\d+\.?\d*)", user_input, re.IGNORECASE)
    cuota_2 = float(cuota_2_match.group(1)) if cuota_2_match else None

    # Construir diccionario de entidades
    entities = {
        "local_team": local_team,
        "visitor_team": visitor_team,
        "local_players": local_players,
        "visitor_players": visitor_players,
        "referee": referee,
        "competition": competition,
        "match_date": match_date,
        "cuota_1": cuota_1,
        "cuota_x": cuota_x,
        "cuota_2": cuota_2
    }

    return entities

def select_model(entities: Dict[str, Any]) -> str:
    """
    Determina qué modelo utilizar según las entidades extraídas.
    """
    # Verificar si tenemos los equipos básicos
    if not entities["local_team"] or not entities["visitor_team"]:
        raise ValueError("No se pudieron identificar los equipos en el mensaje")

    # Verificar si tenemos información adicional
    has_additional_info = (
        entities["local_players"] or
        entities["visitor_players"] or
        entities["referee"] or
        (entities["competition"] and entities["competition"] != "First Division") or
        entities["cuota_1"] is not None or
        entities["cuota_x"] is not None or
        entities["cuota_2"] is not None
    )

    # Seleccionar modelo según la información disponible
    if has_additional_info:
        return "model5"
    else:
        return "model1"

# --------------------------
# Formateo de respuestas
# --------------------------

def format_model1_response(
    response: Dict[str, Any],
    local_team: str,
    visitor_team: str
) -> str:
    """
    Formatea la respuesta del modelo 1 de manera amigable.
    """
    # Capitalizar nombres de equipos para mejor presentación
    local_team_cap = local_team.title()
    visitor_team_cap = visitor_team.title()

    # Determinar el ganador según la respuesta
    winner = response.get("winner", "")

    if winner == "local":
        winning_team = local_team_cap
    elif winner == "visitor":
        winning_team = visitor_team_cap
    else:
        # Si por alguna razón no hay un ganador claro, usar probabilidades
        prob_local = response.get("prob_local", 0)
        prob_visitor = response.get("prob_visitor", 0)

        if prob_local > prob_visitor:
            winning_team = local_team_cap
        elif prob_visitor > prob_local:
            winning_team = visitor_team_cap
        else:
            return f"Basado en mi análisis, el partido entre {local_team_cap} y {visitor_team_cap} será muy parejo, sin un claro favorito."

    # Formatear mensaje amigable
    return f"Según mi análisis, {winning_team} tiene más probabilidades de ganar el partido contra {visitor_team_cap if winning_team == local_team_cap else local_team_cap}."

def format_model5_response(
    response: Dict[str, Any],
    local_team: str,
    visitor_team: str,
    competition: Optional[str] = None
) -> str:
    """
    Formatea la respuesta del modelo 5 de manera detallada y amigable.
    """
    # Capitalizar nombres de equipos para mejor presentación
    local_team_cap = local_team.title()
    visitor_team_cap = visitor_team.title()

    # Obtener probabilidades
    prob_local = response.get("prob_local", 0)
    prob_visitor = response.get("prob_visitor", 0)
    prob_draw = response.get("prob_draw", 0)

    # Convertir a porcentajes
    prob_local_pct = round(prob_local * 100, 1)
    prob_visitor_pct = round(prob_visitor * 100, 1)
    prob_draw_pct = round(prob_draw * 100, 1)

    # Determinar el resultado más probable
    winner = response.get("winner", "")

    # Construir mensaje base
    if competition:
        message = f"Análisis detallado del partido de {competition} entre {local_team_cap} y {visitor_team_cap}:\n\n"
    else:
        message = f"Análisis detallado del partido entre {local_team_cap} y {visitor_team_cap}:\n\n"

    # Añadir probabilidades
    message += f"Probabilidades:\n"
    message += f"• Victoria de {local_team_cap}: {prob_local_pct}%\n"
    message += f"• Empate: {prob_draw_pct}%\n"
    message += f"• Victoria de {visitor_team_cap}: {prob_visitor_pct}%\n\n"

    # Añadir conclusión
    if winner == local_team_cap:
        message += f"Conclusión: {local_team_cap} es favorito para ganar este encuentro."
    elif winner == visitor_team_cap:
        message += f"Conclusión: {visitor_team_cap} es favorito para ganar este encuentro."
    elif winner == "Empate":
        message += f"Conclusión: El partido tiene altas probabilidades de terminar en empate."
    else:
        # Si no hay un ganador claro en la respuesta
        if prob_local > prob_visitor and prob_local > prob_draw:
            message += f"Conclusión: {local_team_cap} es favorito para ganar este encuentro."
        elif prob_visitor > prob_local and prob_visitor > prob_draw:
            message += f"Conclusión: {visitor_team_cap} es favorito para ganar este encuentro."
        else:
            message += f"Conclusión: El partido tiene altas probabilidades de terminar en empate."

    # Añadir información adicional
    confidence = response.get("confidence", 0)
    confidence_pct = round(confidence * 100, 1)

    if confidence_pct > 60:
        message += f" La confianza en esta predicción es alta ({confidence_pct}%)."
    elif confidence_pct > 40:
        message += f" La confianza en esta predicción es moderada ({confidence_pct}%)."
    else:
        message += f" La confianza en esta predicción es baja ({confidence_pct}%), sugiriendo un partido muy parejo."

    return message

# --------------------------
# Herramientas del agente
# --------------------------

@tool
def predict_match_result(match_input: str) -> str:
    """
    Predice el resultado de un partido de fútbol basado en la información proporcionada.
    """
    try:
        # --- 1. Extraer entidades del texto ---
        entities = extract_all_entities(match_input)

        # Asegurarse de que tenemos los dos equipos
        if not entities["local_team"] or not entities["visitor_team"]:
            return ("No pude identificar claramente los equipos en tu mensaje. "
                    "Por favor, menciona los nombres de los dos equipos que jugarán.")

        # --- 2. Construir payload para el router ---
        payload = {
            "local":   entities["local_team"].title(),
            "visitor": entities["visitor_team"].title(),
            # Si el usuario no dio fecha, ponemos hoy para que cumpla el esquema
            "date":    entities["match_date"] or date.today().isoformat(),
            "referee": entities["referee"],
            "competition": entities["competition"],
            "local_players": entities["local_players"],
            "visitor_players": entities["visitor_players"],
            "cuota_1": entities["cuota_1"],
            "cuota_x": entities["cuota_x"],
            "cuota_2": entities["cuota_2"],
        }
        # Quitar claves vacías / None
        payload = {k: v for k, v in payload.items() if v not in (None, [], "")}

        # --- 3. Llamar al router ---
        print("🔍 [DEBUG] Enviando a router payload:", json.dumps(payload, ensure_ascii=False))
        resp = requests.post(ROUTER_URL, json=payload, timeout=10)
        resp.raise_for_status()
        model_resp = resp.json()
        print("🔍 [DEBUG] Router respondió:", model_resp)
 

        # Saber qué modelo respondió
        routed_to = model_resp.pop("routed_to", "unknown")
        print(f"🔍 [DEBUG] Routed to → {routed_to}")
 
         # --- 4. Formatear la respuesta según el modelo ---
        if routed_to.startswith("model1"):
            return format_model1_response(
                response=model_resp,
                local_team=entities["local_team"],
                visitor_team=entities["visitor_team"],
            )
        else:  # model2 / model5
            return format_model5_response(
                response=model_resp,
                local_team=entities["local_team"],
                visitor_team=entities["visitor_team"],
                competition=entities["competition"],
            )

    except Exception as e:
        return f"Lo siento, ocurrió un error al procesar tu solicitud: {str(e)}"

# --------------------------
# Configuración del agente
# --------------------------

def create_football_agent():
    """
    Crea un agente de LangChain para predicción de resultados de fútbol.
    """
    # Creamos un CallbackManager que vuelca cada mensaje por consola
    callback_manager = CallbackManager([StdOutCallbackHandler()])

    # Crear el modelo LLM con verbose y nuestro callback para debug
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        openai_api_key=OPENAI_API_KEY,
        verbose=True,
        callback_manager=callback_manager
    )

    # Definir el prompt del sistema
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
         Eres un agente experto en fútbol español.

         Tu tarea es analizar mensajes de usuarios y predecir resultados de partidos de fútbol.

         Reglas importantes:
         1. Si el usuario menciona dos equipos, utilizarás una herramienta para predecir el resultado.
         2. Si el usuario proporciona información adicional como jugadores, árbitro, etc., la predicción será más detallada.
         3. Si el usuario menciona solo un equipo o ninguno, pídele amablemente que especifique los dos equipos que jugarán.
         4. Si el usuario pregunta sobre otros temas no relacionados con predicciones de fútbol, responde amablemente que estás especializado en predicciones de partidos.

         Usa un tono conversacional y amigable en tus respuestas.
         """),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Crear las herramientas
    tools = [predict_match_result]

    # Configurar la memoria para mantener el historial de conversación
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Crear el agente
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

    return agent_executor

# --------------------------
# Interfaz de chat
# --------------------------

def chat_with_agent(user_input: str, agent_executor=None, thread_id: str = "default"):
    """
    Función para interactuar con el agente.
    """
    if agent_executor is None:
        agent_executor = create_football_agent()

    # Invocar al agente
    response = agent_executor.invoke({"input": user_input})

    # Extraer y devolver la respuesta
    return response.get("output", "")

# --------------------------
# Chatear con agente
# --------------------------

# Crear el agente
football_agent = create_football_agent()