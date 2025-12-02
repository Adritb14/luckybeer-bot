import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Cliente OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Cargar beers.json al arrancar
with open("beers.json", "r", encoding="utf-8") as f:
    BEERS_DATA = json.load(f)

class ChatRequest(BaseModel):
    message: str

SYSTEM_PROMPT = (
    "Eres el sommelier oficial de cervezas del restaurante Lucky Luke. "
    "Usa SIEMPRE el JSON proporcionado para recomendar cervezas. "
    "Reglas importantes: "
    "1) Si el cliente menciona un estilo claro (ej. 'rubia suave', 'rubia', 'suave', 'IPA', 'tostada', etc.), NO hagas ninguna pregunta. "
    "2) Si el cliente pide 'otra', 'qué más tienes' o similar, NO vuelvas a preguntar: simplemente ofrece otra opción del mismo estilo filtrando por el JSON. "
    "3) SOLO haz una pregunta si la solicitud es totalmente ambigua (p. ej.: 'no sé', 'qué tienes'). "
    "4) Responde siempre en máximo 2 frases, muy directo. "
    "5) Prioriza: estilo → suavidad/intensidad → ABV → sabor. "
    "6) Nunca repitas la misma recomendación dos veces seguidas. "
    "7) Cuando respondas, da siempre 1 recomendación + 1 alternativa del mismo estilo si existe."
)

@app.get("/")
async def root():
    return {"message": "LuckyBeer Bot API funcionando correctamente 🍺🔥🤠"}

@app.post("/api/chat")
async def chat(body: ChatRequest):

    beers_json_text = json.dumps(BEERS_DATA, ensure_ascii=False)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": "Este es el catálogo completo de cervezas en formato JSON, úsalo como única fuente de verdad:\n"
                           + beers_json_text
            },
            {"role": "user", "content": body.message},
        ],
    )

    reply_text = response.output_text
    return {"reply": reply_text}
