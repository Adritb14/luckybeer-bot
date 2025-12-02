import os
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# --- RUTAS BASE ---
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
BEERS_PATH = BASE_DIR / "beers.json"

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("luckybeer-bot")

app = FastAPI(title="LuckyBeer Bot")

# --- CORS (para que tu carta HTML pueda llamar al backend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Si quieres, luego restringimos dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATIC (solo si existe la carpeta) ---
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    logger.info(f"Carpeta static montada en: {STATIC_DIR}")
else:
    logger.warning(f"Carpeta static NO encontrada en: {STATIC_DIR}")

# --- RUTA RAÍZ: sirve index.html si existe ---
@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path))
    return JSONResponse(
        {"message": "Backend LuckyBeer-Bot funcionando. Falta static/index.html."}
    )

# --- CLIENTE OPENAI (lazy para no romper al arrancar si falta la API key) ---
client: OpenAI | None = None

def get_openai_client() -> OpenAI:
    global client
    if client is not None:
        return client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY no está configurada en el entorno")
        raise RuntimeError("OPENAI_API_KEY no está configurada en el servidor")

    client = OpenAI(api_key=api_key)
    logger.info("Cliente OpenAI inicializado correctamente")
    return client

# --- CARGAR beers.json ---
try:
    if BEERS_PATH.is_file():
        with open(BEERS_PATH, "r", encoding="utf-8") as f:
            BEERS_DATA = json.load(f)
        logger.info(f"beers.json cargado correctamente ({len(BEERS_DATA)} cervezas)")
    else:
        logger.warning(f"beers.json NO encontrado en: {BEERS_PATH}")
        BEERS_DATA = []
except Exception as e:
    logger.exception(f"Error cargando beers.json: {e}")
    BEERS_DATA = []

# --- MODELO Pydantic ---
class ChatRequest(BaseModel):
    message: str

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = (
    "Eres el sommelier oficial de cervezas del restaurante Lucky Luke. "
    "Usa SIEMPRE el JSON proporcionado para recomendar cervezas.\n\n"
    "REGLAS IMPORTANTES:\n"
    "1) Si el cliente menciona un estilo claro (ej. 'rubia suave', 'rubia', "
    "'suave', 'IPA', 'tostada', 'negra', 'afrutada', etc.), NO hagas preguntas: "
    "recomienda directamente.\n"
    "2) Si el cliente dice 'otra', 'qué más tienes', 'dame otra similar' o parecido, "
    "NO vuelvas a preguntar: ofrece otra opción del mismo estilo o similar usando el JSON.\n"
    "3) SOLO haz una pregunta si la solicitud es totalmente ambigua (p. ej.: "
    "'no sé', 'sorpréndeme' sin más contexto).\n"
    "4) Responde siempre en máximo 2 frases, muy directo y con un tono canalla pero amable.\n"
    "5) Prioriza: estilo → suavidad/intensidad → ABV → sabor.\n"
    "6) Nunca repitas la misma recomendación dos veces seguidas.\n"
    "7) Cuando respondas, da siempre 1 recomendación principal + 1 alternativa del mismo estilo "
    "o lo más parecida posible.\n"
    "8) No inventes cervezas que no estén en el JSON.\n"
)

# --- ENDPOINT DE CHAT ---
@app.post("/api/chat")
async def chat(body: ChatRequest):
    if not BEERS_DATA:
        # No tumba el servidor, pero avisa claramente
        raise HTTPException(
            status_code=500,
            detail="El catálogo de cervezas (beers.json) no está cargado en el servidor.",
        )

    try:
        client = get_openai_client()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Adjuntamos catálogo como contexto
    beers_json_text = json.dumps(BEERS_DATA, ensure_ascii=False)

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "system",
                    "content": "Este es el catálogo completo de cervezas en formato JSON, "
                               "úsalo como única fuente de verdad:\n"
                               + beers_json_text,
                },
                {"role": "user", "content": body.message},
            ],
        )

        # Usamos la propiedad cómoda de la SDK nueva
        reply_text = getattr(response, "output_text", None)
        if not reply_text:
            # Fallback por si cambia la estructura en el futuro
            try:
                reply_text = response.output[0].content[0].text
            except Exception:
                reply_text = "Ahora mismo no puedo responder, mi barril de IA está vacío."

        return {"reply": reply_text}

    except Exception as e:
        logger.exception(f"Error llamando a OpenAI: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno hablando con el modelo de IA.",
        )

# --- EJECUCIÓN DIRECTA (para probar sin PM2) ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
