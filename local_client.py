"""
ESCANEO — Cliente local de prueba (webcam + OpenCV)
=====================================================

Se conecta al mismo endpoint WebSocket (/camara) que usa la app web,
pero en vez de un navegador, abre tu webcam directo desde VS Code y
muestra los resultados en una ventana de OpenCV. Útil para probar el
modelo rápido, sin pelear con permisos de cámara del navegador.

CÓMO USARLO:
    1. Corre el servidor en otra terminal:  uvicorn main:app --reload
    2. Corre este script:                    python local_client.py
    3. Presiona 'q' en la ventana de video para salir.

Si tu backend corre en otro host/puerto (o ya está en Render), cambia
la variable SERVIDOR_WS más abajo.
"""

import asyncio
import base64
import json

import cv2
import numpy as np
import websockets

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
SERVIDOR_WS = "ws://127.0.0.1:8000/camara"   # local. Para Render: "wss://tu-backend.onrender.com/camara"
INDICE_CAMARA = 0                             # 0 = webcam por defecto. Prueba 1 o 2 si tienes varias cámaras.
UMBRAL_CONFIANZA = 0.30
CALIDAD_JPEG = 70                             # 0-100, más bajo = envío más rápido, menos calidad
ANCHO_ENVIO = 640                             # redimensiona antes de enviar, para que vaya más fluido


async def correr_cliente():
    print(f"Conectando a {SERVIDOR_WS} ...")

    cap = cv2.VideoCapture(INDICE_CAMARA)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara. Prueba cambiar INDICE_CAMARA a 1 o 2.")
        return

    try:
        async with websockets.connect(SERVIDOR_WS, max_size=None) as ws:
            print(" Conectado. Presiona 'q' en la ventana de video para salir.\n")

            while True:
                ok, frame = cap.read()
                if not ok:
                    print("No se pudo leer un frame de la cámara.")
                    break

                # Redimensionar antes de enviar (más rápido, el backend igual
                # redimensiona a 416x416 para el modelo)
                alto_original, ancho_original = frame.shape[:2]
                escala = ANCHO_ENVIO / ancho_original
                frame_envio = cv2.resize(frame, (ANCHO_ENVIO, int(alto_original * escala)))

                # Codificar a JPEG -> base64 (mismo formato que espera el backend)
                ok_enc, buffer = cv2.imencode(
                    '.jpg', frame_envio, [int(cv2.IMWRITE_JPEG_QUALITY), CALIDAD_JPEG]
                )
                if not ok_enc:
                    continue

                imagen_b64 = base64.b64encode(buffer).decode('utf-8')

                mensaje = json.dumps({
                    "imagen": imagen_b64,
                    "umbral": UMBRAL_CONFIANZA
                })

                await ws.send(mensaje)
                respuesta_raw = await ws.recv()
                respuesta = json.loads(respuesta_raw)

                # El backend ya devuelve el frame CON las cajas dibujadas
                imagen_procesada_b64 = respuesta.get("imagen")
                detecciones = respuesta.get("detecciones", [])

                if imagen_procesada_b64:
                    img_bytes = base64.b64decode(imagen_procesada_b64)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    frame_mostrar = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    # Overlay simple con la cuenta de detecciones, en la esquina
                    texto = f"Detecciones: {len(detecciones)}"
                    cv2.putText(
                        frame_mostrar, texto, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

                    cv2.imshow("ESCANEO - Prueba local (presiona 'q' para salir)", frame_mostrar)

                # Log en consola de lo detectado
                if detecciones:
                    resumen = ", ".join(f"{d['objeto']} ({d['confianza']:.2f})" for d in detecciones)
                    print(f"Detectado: {resumen}")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except (ConnectionRefusedError, OSError):
        print(f"❌ No se pudo conectar a {SERVIDOR_WS}. ¿Está corriendo el servidor (uvicorn main:app --reload)?")
    except websockets.exceptions.ConnectionClosed:
        print("La conexión con el servidor se cerró.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Cámara y ventanas cerradas.")


if __name__ == "__main__":
    asyncio.run(correr_cliente())
