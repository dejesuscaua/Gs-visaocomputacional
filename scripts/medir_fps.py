"""Mede o FPS do modelo na CPU, como no computador do morador.

    python scripts/medir_fps.py --fonte video.mp4 --modelo modelo/best.pt
    python scripts/medir_fps.py --fonte 0              # webcam, 200 quadros

Grava resultados/fps.json com FPS médio e latência p50/p95 por tamanho de entrada.
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", default="0")
    p.add_argument("--modelo", default="modelo/best.pt")
    p.add_argument("--quadros", type=int, default=200)
    p.add_argument("--tamanhos", default="640,480,320")
    args = p.parse_args()

    cap = cv2.VideoCapture(int(args.fonte) if args.fonte.isdigit() else args.fonte)
    quadros = []
    while len(quadros) < args.quadros:
        ok, q = cap.read()
        if not ok:
            break
        quadros.append(q)
    cap.release()
    if not quadros:
        raise SystemExit("Nenhum quadro lido da fonte.")

    modelo = YOLO(args.modelo)
    resultado = {"cpu": platform.processor() or platform.machine(), "quadros": len(quadros), "por_tamanho": {}}
    for imgsz in map(int, args.tamanhos.split(",")):
        modelo.predict(quadros[0], imgsz=imgsz, device="cpu", verbose=False)  # aquecimento
        tempos = []
        for q in quadros:
            t = time.perf_counter()
            modelo.predict(q, imgsz=imgsz, device="cpu", verbose=False)
            tempos.append(time.perf_counter() - t)
        resultado["por_tamanho"][imgsz] = {
            "fps_medio": round(len(tempos) / sum(tempos), 1),
            "latencia_p50_ms": round(statistics.median(tempos) * 1000, 1),
            "latencia_p95_ms": round(sorted(tempos)[int(0.95 * (len(tempos) - 1))] * 1000, 1),
        }
        print(imgsz, resultado["por_tamanho"][imgsz])
    Path("resultados").mkdir(exist_ok=True)
    Path("resultados/fps.json").write_text(json.dumps(resultado, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
