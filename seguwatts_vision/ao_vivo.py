"""Processamento quadro a quadro para uso interativo (webcam no Colab ou outra fonte).

O detector de linha de comando (detector.py) lê uma câmera ou um vídeo com o OpenCV.
No Colab a câmera fica no navegador, então os quadros chegam um a um por JavaScript.
Esta classe recebe cada quadro, roda os modelos, aplica a regra de desperdício,
publica os eventos e devolve o quadro anotado com o mesmo painel do detector.
"""

from __future__ import annotations

import time

import numpy as np
from ultralytics import YOLO

from seguwatts_vision.config import Config
from seguwatts_vision.detector import COCO_PESSOA, classes_do_modelo, desenhar, extrair_deteccoes
from seguwatts_vision.eventos import SaidaEventos
from seguwatts_vision.regra import MonitorDesperdicio, Resultado


class SessaoAoVivo:
    def __init__(
        self,
        modelo: str = "modelo/best.pt",
        modelo_pessoas: str = "yolo11n.pt",
        comodo: str = "sala",
        limite_s: float = 10.0,
        imgsz: int = 640,
        arquivo_eventos: str | None = None,
    ) -> None:
        cfg = Config()
        if arquivo_eventos:
            cfg.arquivo_eventos = arquivo_eventos
        self.cfg = cfg
        self.imgsz = imgsz
        self.modelo = YOLO(modelo)
        conhecidas = classes_do_modelo(self.modelo)
        self.modelo_pessoas = None if "pessoa" in conhecidas else YOLO(modelo_pessoas)
        self.monitor = MonitorDesperdicio(comodo, limite_s=limite_s, janela_s=cfg.janela_s,
                                          confianca_min=cfg.confianca_min)
        self.saida = SaidaEventos(cfg.arquivo_eventos, cfg.tarifa_rs_kwh, cfg.webhook_url)
        self.eventos: list[dict] = []
        self._ultimo = time.perf_counter()
        self.fps = 0.0

    def processar(self, quadro: np.ndarray, agora: float | None = None) -> tuple[np.ndarray, Resultado]:
        """Recebe um quadro BGR, devolve o mesmo quadro anotado e o estado do cômodo."""
        agora = time.time() if agora is None else agora
        deteccoes = extrair_deteccoes(self.modelo(quadro, imgsz=self.imgsz, conf=0.25, verbose=False)[0])
        if self.modelo_pessoas is not None:
            pessoas = self.modelo_pessoas(quadro, imgsz=self.imgsz, conf=0.25, classes=[COCO_PESSOA], verbose=False)[0]
            deteccoes += extrair_deteccoes(pessoas)

        r = self.monitor.atualizar(deteccoes, agora)
        for evento in (r.evento_aberto, r.evento_encerrado):
            if evento is not None:
                self.eventos.append(self.saida.publicar(evento))

        t = time.perf_counter()
        inst = 1.0 / max(t - self._ultimo, 1e-6)
        self.fps = inst if self.fps == 0 else 0.8 * self.fps + 0.2 * inst
        self._ultimo = t

        anotado = quadro.copy()
        desenhar(anotado, deteccoes, r, self.monitor, self.fps, self.cfg.tarifa_rs_kwh)
        return anotado, r

    def encerrar(self, agora: float | None = None) -> None:
        evento = self.monitor.finalizar(time.time() if agora is None else agora)
        if evento is not None:
            self.eventos.append(self.saida.publicar(evento))
