import base64
import json
from pathlib import Path

import gdown
import cv2
import numpy as np
import keras
import keras_cv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


# =====================================
# MODELO
# =====================================

CARPETA_MODELO = Path(__file__).parent

RUTA_CONFIG = CARPETA_MODELO / "config.json"
RUTA_PESOS = CARPETA_MODELO / "model.weights.h5"


ID_MODELO = "1QWpWvCz5ywtYjDyaUoLMygLb5kchewQl"


if not RUTA_PESOS.exists():

    print("Descargando modelo...")

    gdown.download(
        id=ID_MODELO,
        output=str(RUTA_PESOS),
        quiet=False
    )


print("Cargando modelo YOLO...")


with open(RUTA_CONFIG,"r",encoding="utf-8") as f:
    config_modelo=json.load(f)


modelo = keras.saving.deserialize_keras_object(
    config_modelo
)

modelo.load_weights(
    str(RUTA_PESOS)
)


print("Modelo listo")


# =====================================
# CONFIG
# =====================================

TAMANO_ENTRADA=416


NOMBRES_CLASES=[
"Backpack",
"Bed",
"Bottle",
"Chair",
"Couch",
"Door",
"Fork",
"Glass",
"Hat",
"Jug",
"Knife",
"Lamp",
"Mirror",
"Mug",
"Oven",
"Plate",
"Spoon",
"Table",
"Television",
"Wok",
]


# =====================================
# FASTAPI
# =====================================


app=FastAPI(
    title="ESCANEO IA"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# =====================================
# BASE64
# =====================================


def decodificar_imagen_b64(data):

    img_bytes=base64.b64decode(data)

    arr=np.frombuffer(
        img_bytes,
        np.uint8
    )

    return cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR
    )



# =====================================
# YOLO
# =====================================


def correr_inferencia(frame_bgr,umbral):


    alto,ancho=frame_bgr.shape[:2]


    escala=min(
        TAMANO_ENTRADA/ancho,
        TAMANO_ENTRADA/alto
    )


    nw=int(ancho*escala)
    nh=int(alto*escala)


    rgb=cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB
    )


    resized=cv2.resize(
        rgb,
        (nw,nh)
    )


    pad_x=TAMANO_ENTRADA-nw
    pad_y=TAMANO_ENTRADA-nh


    letter=cv2.copyMakeBorder(
        resized,
        0,
        pad_y,
        0,
        pad_x,
        cv2.BORDER_CONSTANT,
        value=(0,0,0)
    )


    entrada=np.expand_dims(
        letter.astype(np.float32)/255.0,
        axis=0
    )


    pred=modelo.predict(
        entrada,
        verbose=0
    )


    cajas=np.array(
        pred["boxes"][0]
    )

    confs=np.array(
        pred["confidence"][0]
    )

    clases=np.array(
        pred["classes"][0]
    ).astype(int)


    cantidad=int(
        pred["num_detections"][0]
    )



    candidatos=[]


    for i in range(cantidad):

        conf=float(confs[i])


        if conf < umbral:
            continue


        clase=clases[i]


        if clase<0 or clase>=len(NOMBRES_CLASES):
            continue



        x1,y1,x2,y2=cajas[i]


        x1=(x1*TAMANO_ENTRADA)/escala
        y1=(y1*TAMANO_ENTRADA)/escala
        x2=(x2*TAMANO_ENTRADA)/escala
        y2=(y2*TAMANO_ENTRADA)/escala



        x1=int(max(0,min(x1,ancho)))
        y1=int(max(0,min(y1,alto)))
        x2=int(max(0,min(x2,ancho)))
        y2=int(max(0,min(y2,alto)))


        candidatos.append(
            (
                x1,
                y1,
                x2,
                y2,
                conf,
                clase
            )
        )



    detecciones=[]


    for x1,y1,x2,y2,conf,clase in candidatos:


        detecciones.append({

            "objeto":
                NOMBRES_CLASES[clase],

            "confianza":
                round(conf,3),

            "x":
                x1,

            "y":
                y1,

            "ancho":
                x2-x1,

            "alto":
                y2-y1

        })



    return detecciones




# =====================================
# WEBSOCKET
# =====================================


@app.websocket("/camara")
async def camara(ws:WebSocket):


    await ws.accept()


    print("Cliente conectado")


    try:


        while True:


            mensaje=await ws.receive_text()


            data=json.loads(
                mensaje
            )


            imagen=data.get(
                "imagen"
            )


            umbral=float(
                data.get(
                    "umbral",
                    0.55
                )
            )


            if not imagen:
                continue



            frame=decodificar_imagen_b64(
                imagen
            )


            if frame is None:
                continue



            detecciones=correr_inferencia(
                frame,
                umbral
            )



            await ws.send_text(
                json.dumps({

                    "detecciones":
                        detecciones

                })
            )



    except WebSocketDisconnect:

        print("Cliente desconectado")


    except Exception as e:

        print(
            "Error:",
            e
        )



@app.get("/")
async def inicio():

    return {

        "estado":"ok",

        "mensaje":
        "Detector funcionando"

    }
