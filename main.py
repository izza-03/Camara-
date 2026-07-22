"""
ESCANEO — Servidor backend (FastAPI + WebSocket)
=================================================

Carga el modelo YOLOv8 (Keras-CV) entrenado y expone un endpoint
WebSocket /camara compatible con local_client.py y con una futura
app web.

Protocolo (lo mismo que espera local_client.py):
    Cliente -> Servidor:
        {"imagen": "<jpg en base64>", "umbral": 0.30}
    Servidor -> Cliente:
        {"imagen": "<jpg anotado en base64>",
         "detecciones": [{"objeto": "Taza", "confianza": 0.87}, ...]}

CÓMO USARLO:
    1. Coloca este archivo, model.weights.h5 y config.json en la misma carpeta.
    2. Activa el venv:      C:\\Users\\lurjz\\venvs\\objdetector\\Scripts\\Activate.ps1
    3. Corre el servidor:   uvicorn main:app --reload
    4. En otra terminal (con el mismo venv activo), corre: python local_client.py
"""

import base64
import json
from pathlib import Path

import gdown

import cv2
import numpy as np
import keras
import keras_cv  # noqa: F401  (necesario para registrar las clases custom de Keras-CV)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


# Config

CARPETA_MODELO = Path(__file__).parent
RUTA_CONFIG = CARPETA_MODELO / "config.json"
RUTA_PESOS = CARPETA_MODELO / "model.weights.h5"


# Descargar modelo desde Google Drive si no existe

ID_MODELO = "1QWpWvCz5ywtYjDyaUoLMygLb5kchewQl"

if not RUTA_PESOS.exists():
    print("Descargando model.weights.h5 desde Google Drive...")

    gdown.download(
        id=ID_MODELO,
        output=str(RUTA_PESOS),
        quiet=False
    )

    print("Modelo descargado correctamente.")


TAMANO_ENTRADA = 416  # el modelo acepta cualquier tamaño (input_shape=[null,null,3]),
                       # pero entrenamos/inferimos a 416x416

NOMBRES_CLASES = [
    "Backpack", "Bed", "Bottle", "Chair", "Couch", "Door", "Fork", "Glass",
    "Hat", "Jug", "Knife", "Lamp", "Mirror", "Mug", "Oven", "Plate",
    "Spoon", "Table", "Television", "Wok",
]

# ---------------------------------------------------------------
# Cargar el modelo una sola vez al arrancar el servidor
# ---------------------------------------------------------------
print("Cargando modelo YOLOv8 (Keras-CV) ...")

with open(RUTA_CONFIG, "r", encoding="utf-8") as f:
    config_modelo = json.load(f)

modelo = keras.saving.deserialize_keras_object(config_modelo)
modelo.load_weights(str(RUTA_PESOS))

print("Modelo cargado correctamente.")

# ---------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------
app = FastAPI(title="ESCANEO - Detector de objetos")

# Permite que una futura app web (otro origen) se conecte sin problemas de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def decodificar_imagen_b64(imagen_b64: str) -> np.ndarray:
    """base64 -> imagen BGR (formato OpenCV)."""
    img_bytes = base64.b64decode(imagen_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def codificar_imagen_b64(frame_bgr: np.ndarray, calidad: int = 70) -> str:
    """imagen BGR -> base64 (jpg)."""
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), calidad])
    if not ok:
        raise ValueError("No se pudo codificar la imagen a JPEG.")
    return base64.b64encode(buffer).decode("utf-8")


def correr_inferencia(frame_bgr: np.ndarray, umbral: float):

    alto_original, ancho_original = frame_bgr.shape[:2]

    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB
    )

    frame_resized = cv2.resize(
        frame_rgb,
        (TAMANO_ENTRADA, TAMANO_ENTRADA)
    )

    batch = np.expand_dims(
        frame_resized.astype(np.float32),
        axis=0
    )


    prediccion = modelo.predict(
        batch,
        verbose=0
    )


    # Salida actual del modelo
    cajas = np.array(
        prediccion["boxes"][0]
    )

    clases = np.array(
        prediccion["classes"][0]
    ).astype(int)


    # Como el modelo actual no entrega confidence
    confianzas = np.ones(
        len(clases),
        dtype=np.float32
    )


    cantidad = len(clases)


    frame_dibujado = frame_bgr.copy()

    detecciones = []


    for i in range(cantidad):

        conf = float(confianzas[i])


        if conf < umbral:
            continue


        idx_clase = clases[i]


        if idx_clase < 0 or idx_clase >= len(NOMBRES_CLASES):
            continue


        nombre = NOMBRES_CLASES[idx_clase]


        x1, y1, x2, y2 = cajas[i]


        # KerasCV normalmente entrega coordenadas relativas
        x1 = int(x1 * ancho_original)
        y1 = int(y1 * alto_original)
        x2 = int(x2 * ancho_original)
        y2 = int(y2 * alto_original)


        x1 = max(0, min(x1, ancho_original))
        y1 = max(0, min(y1, alto_original))
        x2 = max(0, min(x2, ancho_original))
        y2 = max(0, min(y2, alto_original))


        cv2.rectangle(
            frame_dibujado,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )


        etiqueta = f"{nombre} {conf:.2f}"


        cv2.putText(
            frame_dibujado,
            etiqueta,
            (x1,max(y1-8,15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )


        detecciones.append({

            "objeto": nombre,

            "confianza": round(conf,3),

            "x": x1,

            "y": y1,

            "ancho": x2-x1,

            "alto": y2-y1

        })


    return frame_dibujado, detecciones



@app.websocket("/camara")
async def endpoint_camara(websocket: WebSocket):
    await websocket.accept()
    print("Cliente conectado.")

    try:
        while True:
            mensaje_raw = await websocket.receive_text()
            mensaje = json.loads(mensaje_raw)

            imagen_b64 = mensaje.get("imagen")
            umbral = float(mensaje.get("umbral", 0.30))

            if not imagen_b64:
                continue

            frame = decodificar_imagen_b64(imagen_b64)
            if frame is None:
                continue

            frame_procesado, detecciones = correr_inferencia(frame, umbral)
            imagen_salida_b64 = codificar_imagen_b64(frame_procesado)

            await websocket.send_text(json.dumps({
                "imagen": imagen_salida_b64,
                "detecciones": detecciones,
            }))

    except WebSocketDisconnect:
        print("Cliente desconectado.")
    except Exception as e:
        print(f"Error en el WebSocket: {e}")


@app.get("/")
async def raiz():
    return {"estado": "ok", "mensaje": "Servidor ESCANEO corriendo. Endpoint WebSocket en /camara"}
