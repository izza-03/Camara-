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

import cv2
import numpy as np
import keras
import keras_cv  # noqa: F401  (necesario para registrar las clases custom de Keras-CV)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
CARPETA_MODELO = Path(__file__).parent
RUTA_CONFIG = CARPETA_MODELO / "config.json"
RUTA_PESOS = CARPETA_MODELO / "model.weights.h5"

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
    """
    Corre el modelo sobre un frame y devuelve:
        - el mismo frame con las cajas dibujadas (tamaño original)
        - la lista de detecciones [{"objeto": ..., "confianza": ...}, ...]
    """
    alto_original, ancho_original = frame_bgr.shape[:2]

    # --- Preprocesar con "letterbox": redimensionar manteniendo la proporción
    # original y rellenar con bordes negros hasta llegar a un cuadrado de
    # TAMANO_ENTRADA x TAMANO_ENTRADA. Un resize directo (sin esto) ESTIRA la
    # imagen (ej. de 16:9 a 1:1) y deforma los objetos, lo que hace que el
    # modelo saque cajas erráticas por todos lados.
    escala = min(TAMANO_ENTRADA / ancho_original, TAMANO_ENTRADA / alto_original)
    ancho_nuevo = int(round(ancho_original * escala))
    alto_nuevo = int(round(alto_original * escala))

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_reescalado = cv2.resize(frame_rgb, (ancho_nuevo, alto_nuevo))

    relleno_x = TAMANO_ENTRADA - ancho_nuevo
    relleno_y = TAMANO_ENTRADA - alto_nuevo
    # el padding se pone abajo/derecha, así el origen (0,0) no se mueve
    frame_letterbox = cv2.copyMakeBorder(
        frame_reescalado, 0, relleno_y, 0, relleno_x,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # NOTA: en el notebook de entrenamiento se dividió manualmente entre 255
    # ANTES de pasar la imagen al modelo, pero el backbone ya tiene
    # include_rescaling=True (osea, vuelve a dividir entre 255 él solo).
    # Esto significa que el modelo entrenó viendo imágenes en el rango
    # [0, 1/255] en vez de [0, 1] o [0, 255]. Para que la inferencia coincida
    # con lo que el modelo realmente aprendió, reproducimos ese mismo
    # doble-escalado aquí (dividir entre 255 antes de mandarlo al modelo).
    batch = np.expand_dims((frame_letterbox.astype(np.float32) / 255.0), axis=0)

    prediccion = modelo.predict(batch, verbose=0)

    cajas = np.array(prediccion["boxes"][0])         # rel_xyxy, relativo al cuadro 416x416 con letterbox
    confianzas = np.array(prediccion["confidence"][0])
    clases = np.array(prediccion["classes"][0]).astype(int)
    num_detecciones = int(prediccion["num_detections"][0])

    frame_dibujado = frame_bgr.copy()
    detecciones = []

    # --- Paso 1: armar la lista de candidatos que pasan el umbral, con sus
    # coordenadas ya reescaladas al frame original.
    candidatos = []  # cada uno: (x1, y1, x2, y2, confianza, idx_clase)
    for i in range(num_detecciones):
        conf = float(confianzas[i])
        if conf < umbral:
            continue

        idx_clase = clases[i]
        if idx_clase < 0 or idx_clase >= len(NOMBRES_CLASES):
            continue

        # rel_xyxy (0-1 sobre los 416x416 con letterbox) -> píxeles sobre los
        # 416x416 -> quitar el padding -> reescalar de vuelta al tamaño original
        x1, y1, x2, y2 = cajas[i]
        x1 = (x1 * TAMANO_ENTRADA) / escala
        y1 = (y1 * TAMANO_ENTRADA) / escala
        x2 = (x2 * TAMANO_ENTRADA) / escala
        y2 = (y2 * TAMANO_ENTRADA) / escala

        x1 = int(max(0, min(x1, ancho_original)))
        y1 = int(max(0, min(y1, alto_original)))
        x2 = int(max(0, min(x2, ancho_original)))
        y2 = int(max(0, min(y2, alto_original)))

        candidatos.append((x1, y1, x2, y2, conf, idx_clase))

    # --- Paso 2: NMS extra, más estricto que el del modelo (iou_threshold=0.7
    # en config.json es muy permisivo y deja pasar cajas casi duplicadas para
    # el mismo objeto). Se aplica por clase, para no descartar objetos
    # distintos que sí se superpongan.
    indices_finales = []
    for idx_clase_actual in set(c[5] for c in candidatos):
        idxs_clase = [i for i, c in enumerate(candidatos) if c[5] == idx_clase_actual]
        cajas_xywh = [
            [candidatos[i][0], candidatos[i][1],
             candidatos[i][2] - candidatos[i][0], candidatos[i][3] - candidatos[i][1]]
            for i in idxs_clase
        ]
        confs_clase = [candidatos[i][4] for i in idxs_clase]

        indices_nms = cv2.dnn.NMSBoxes(
            cajas_xywh, confs_clase, score_threshold=umbral, nms_threshold=0.3
        )
        for j in np.array(indices_nms).flatten().tolist() if len(indices_nms) else []:
            indices_finales.append(idxs_clase[j])

    # --- Paso 3: dibujar solo lo que sobrevivió al NMS extra
    for i in indices_finales:
        x1, y1, x2, y2, conf, idx_clase = candidatos[i]
        nombre = NOMBRES_CLASES[idx_clase]

        cv2.rectangle(frame_dibujado, (x1, y1), (x2, y2), (0, 255, 0), 2)
        etiqueta = f"{nombre} {conf:.2f}"
        cv2.putText(
            frame_dibujado, etiqueta, (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        detecciones.append({"objeto": nombre, "confianza": round(conf, 4)})

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
