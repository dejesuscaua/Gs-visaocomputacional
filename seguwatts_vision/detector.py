"""SeguWatts Vision: detecta aparelho ligado em cômodo vazio, em tempo real.

Uso:
    python -m seguwatts_vision.detector                          # webcam 0, limite de 120 s
    python -m seguwatts_vision.detector --demo                   # limite de 10 s, para o vídeo
    python -m seguwatts_vision.detector --fonte video.mp4 --saida resultados/demo.mp4 --sem-janela

Teclas na janela: q sai, d liga/desliga o modo demonstração (limite de 10 s).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from seguwatts_vision.config import (
    CLASSES_DESLIGADO,
    CLASSES_LIGADO,
    CLASSES_PRESENCA,
    Config,
    estado_tela,
    nome_tem_estado,
    normalizar_classe,
)
from seguwatts_vision.eventos import SaidaEventos
from seguwatts_vision.regra import Deteccao, Estado, MonitorDesperdicio, Resultado

# Paleta SeguWatts em BGR (OpenCV).
MAGENTA = (95, 30, 237)
VINHO = (54, 16, 126)
NOITE = (24, 20, 20)
CINZA = (178, 167, 169)
VERDE = (132, 220, 61)
BRANCO = (245, 242, 243)
FONTE = cv2.FONT_HERSHEY_SIMPLEX
# Hershey não tem acentos: textos da tela sem acentuação de propósito.
NOME_AMIGAVEL = {"lampada_acesa": "luz", "tela_ligada": "TV/monitor"}


def duracao_legivel(segundos: float) -> str:
    if segundos < 60:
        return f"{segundos:.0f} s"
    return f"{int(segundos // 60)} min {int(segundos % 60):02d} s"


def cor_da_classe(classe: str) -> tuple[int, int, int]:
    if classe in CLASSES_LIGADO:
        return MAGENTA
    if classe in CLASSES_PRESENCA:
        return VERDE
    if classe in CLASSES_DESLIGADO:
        return CINZA
    return BRANCO


COCO_PESSOA = 0  # índice da classe "person" no COCO
COCO_TV = 62  # índice da classe "tv" no COCO
COCO_CLASSES = [COCO_PESSOA, COCO_TV]


def classificar_tela(quadro, caixa) -> str:
    """Olha o brilho dentro da caixa da TV (sem a moldura) e decide ligada ou desligada."""
    cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
    h, w = cinza.shape
    x1, y1, x2, y2 = (int(v) for v in caixa)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    mx, my = (x2 - x1) // 10, (y2 - y1) // 10  # descarta 10% de cada borda (moldura)
    tela = cinza[y1 + my:y2 - my, x1 + mx:x2 - mx]
    if tela.size == 0:
        return "tela_desligada"
    return estado_tela(float(tela.mean()), float(tela.std()), float(cinza.mean()))


def extrair_deteccoes(resultado, quadro=None) -> list[Deteccao]:
    """Converte a saída da YOLO em detecções com os nomes do SeguWatts.

    O nome de cada classe passa por normalizar_classe(): "Lamp On" vira lampada_acesa,
    "person" vira pessoa, e classes sem interesse (cadeira, garrafa...) são descartadas.
    Uma TV sem estado no nome ("tv" do COCO) tem o estado decidido pelo brilho da tela,
    quando o quadro é informado.
    """
    nomes = resultado.names
    caixas = resultado.boxes
    if caixas is None or len(caixas) == 0:
        return []
    deteccoes = []
    for c, p, xyxy in zip(caixas.cls.tolist(), caixas.conf.tolist(), caixas.xyxy.tolist()):
        nome = nomes[int(c)]
        classe = normalizar_classe(nome)
        if classe is None:
            continue
        caixa = tuple(map(float, xyxy))
        if classe == "tela_ligada" and quadro is not None and not nome_tem_estado(nome):
            classe = classificar_tela(quadro, caixa)
        deteccoes.append(Deteccao(classe, float(p), caixa))
    return deteccoes


def classes_do_modelo(modelo) -> set[str]:
    return {c for c in (normalizar_classe(n) for n in modelo.names.values()) if c}


def texto(img, msg: str, xy: tuple[int, int], escala: float = 0.6, cor=BRANCO, espessura: int = 1) -> None:
    cv2.putText(img, msg, xy, FONTE, escala, cor, espessura, cv2.LINE_AA)


def desenhar(img, deteccoes: list[Deteccao], r: Resultado, monitor: MonitorDesperdicio, fps: float, tarifa: float) -> None:
    h, w = img.shape[:2]

    for d in deteccoes:
        if d.confianca < monitor.confianca_min:
            continue
        x1, y1, x2, y2 = map(int, d.caixa)
        cor = cor_da_classe(d.classe)
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
        rotulo = f"{d.classe} {d.confianca:.2f}"
        (tw, th), _ = cv2.getTextSize(rotulo, FONTE, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), cor, -1)
        texto(img, rotulo, (x1 + 4, y1 - 5), 0.5, NOITE if cor != MAGENTA else BRANCO)

    # Barra superior: cômodo, estado, FPS.
    cv2.rectangle(img, (0, 0), (w, 40), NOITE, -1)
    texto(img, "SeguWatts Vision", (12, 27), 0.65, MAGENTA, 2)
    estados = {
        Estado.OCUPADO: ("ocupado" if r.presenca else "nada ligado", VERDE if r.presenca else CINZA),
        Estado.VAZIO_LIGADO: ("vazio com aparelho ligado", MAGENTA),
        Estado.DESPERDICIO: ("POSSIVEL DESPERDICIO", MAGENTA),
    }
    nome_estado, cor_estado = estados[r.estado]
    texto(img, f"{monitor.comodo} | {nome_estado}", (210, 27), 0.6, cor_estado, 1)
    texto(img, f"{fps:4.1f} FPS", (w - 100, 27), 0.6, CINZA)

    # Contador até o limite.
    if r.estado is Estado.VAZIO_LIGADO:
        progresso = min(1.0, r.segundos_vazio / monitor.limite_s)
        cv2.rectangle(img, (0, 40), (w, 46), VINHO, -1)
        cv2.rectangle(img, (0, 40), (int(w * progresso), 46), MAGENTA, -1)
        texto(img, f"vazio ha {duracao_legivel(r.segundos_vazio)} (alerta em {monitor.limite_s:.0f} s)", (12, 68), 0.55, BRANCO)

    # Faixa de alerta.
    evento = monitor.evento_atual
    if r.estado is Estado.DESPERDICIO and evento is not None:
        cv2.rectangle(img, (0, h - 64), (w, h), MAGENTA, -1)
        aparelhos = " e ".join(NOME_AMIGAVEL.get(a, a) for a in evento.aparelhos)
        custo = f"{evento.energia_kwh() * tarifa:.4f}".replace(".", ",")
        texto(img, f"Possivel desperdicio: {aparelhos} ligado sem ninguem", (14, h - 38), 0.62, BRANCO, 2)
        texto(img, f"ha {duracao_legivel(r.segundos_vazio)} | estimativa ate agora: R$ {custo}", (14, h - 14), 0.55, BRANCO)


def abrir_fonte(fonte: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(fonte) if fonte.isdigit() else fonte)
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir a fonte de vídeo '{fonte}'. Confira o índice da webcam ou o caminho do arquivo.")
    return cap


def executar(args: argparse.Namespace) -> None:
    cfg = Config()
    limite = 10.0 if args.demo else (args.limite or cfg.limite_s)
    if args.pre_treinado:
        # Só pesos COCO: pessoa e TV (estado da TV pelo brilho da tela), sem lâmpadas.
        logging.warning("modo --pre-treinado: pesos COCO, detecta pessoa e TV, mas não lâmpadas.")
    modelo = YOLO("yolo11n.pt" if args.pre_treinado else args.modelo)
    conhecidas = classes_do_modelo(modelo)
    logging.info("classes do modelo principal: %s", ", ".join(sorted(conhecidas)) or "nenhuma reconhecida")
    if not conhecidas & (CLASSES_LIGADO | CLASSES_DESLIGADO):
        logging.warning("o modelo principal não tem classes de aparelho: nenhum desperdício será detectado")
    # O modelo treinado pela equipe cuida das lâmpadas. Pessoa e TV, que ele não tem
    # (dataset público de lâmpadas), vêm da YOLO pré-treinada no COCO no mesmo quadro.
    modelo_pessoas = None
    if not {"pessoa", "tela_ligada"} <= conhecidas:
        modelo_pessoas = YOLO(args.modelo_pessoas)
        logging.info("pessoas e TVs detectadas por %s (COCO)", args.modelo_pessoas)
    monitor = MonitorDesperdicio(args.comodo or cfg.comodo, limite_s=limite, janela_s=cfg.janela_s, confianca_min=cfg.confianca_min)
    saida = SaidaEventos(cfg.arquivo_eventos, cfg.tarifa_rs_kwh, cfg.webhook_url)

    cap = abrir_fonte(args.fonte)
    eh_arquivo = not args.fonte.isdigit()
    gravador = None
    fps_suave, ultimo = 0.0, time.perf_counter()
    agora = time.time()
    inicio_video = time.time()

    try:
        while True:
            ok, quadro = cap.read()
            if not ok:
                break
            # Em vídeo gravado, o relógio é o do próprio vídeo: a regra funciona igual
            # mesmo que o processamento seja mais rápido ou mais lento que o tempo real.
            agora = inicio_video + cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if eh_arquivo else time.time()

            resultado = modelo(quadro, imgsz=args.imgsz, conf=0.25, verbose=False)[0]
            deteccoes = extrair_deteccoes(resultado, quadro)
            if modelo_pessoas is not None:
                coco = modelo_pessoas(quadro, imgsz=args.imgsz, conf=0.25, classes=COCO_CLASSES, verbose=False)[0]
                deteccoes += extrair_deteccoes(coco, quadro)
            r = monitor.atualizar(deteccoes, agora)
            if r.evento_aberto:
                saida.publicar(r.evento_aberto)
            if r.evento_encerrado:
                saida.publicar(r.evento_encerrado)

            t = time.perf_counter()
            fps_inst = 1.0 / max(t - ultimo, 1e-6)
            fps_suave = fps_inst if fps_suave == 0 else 0.9 * fps_suave + 0.1 * fps_inst
            ultimo = t

            desenhar(quadro, deteccoes, r, monitor, fps_suave, cfg.tarifa_rs_kwh)

            if args.saida:
                if gravador is None:
                    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
                    fps_video = cap.get(cv2.CAP_PROP_FPS) or 20.0
                    h, w = quadro.shape[:2]
                    gravador = cv2.VideoWriter(args.saida, cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (w, h))
                gravador.write(quadro)

            if not args.sem_janela:
                cv2.imshow("SeguWatts Vision", quadro)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
                if tecla == ord("d"):
                    monitor.limite_s = 10.0 if monitor.limite_s != 10.0 else (args.limite or cfg.limite_s)
                    logging.info("limite agora é %.0f s", monitor.limite_s)
    finally:
        encerrado = monitor.finalizar(agora)
        if encerrado:
            saida.publicar(encerrado)
        cap.release()
        if gravador is not None:
            gravador.release()
        if not args.sem_janela:
            cv2.destroyAllWindows()
        logging.info("FPS médio aproximado: %.1f", fps_suave)


def main() -> None:
    p = argparse.ArgumentParser(description="SeguWatts Vision: aparelho ligado em cômodo vazio.")
    p.add_argument("--fonte", default="0", help="índice da webcam (0, 1...) ou caminho de vídeo")
    p.add_argument("--modelo", default="modelo/best.pt", help="pesos YOLO treinados (aparelhos)")
    p.add_argument("--modelo-pessoas", default="yolo11n.pt",
                   help="pesos COCO para detectar pessoas e TVs (o modelo treinado só tem lâmpadas)")
    p.add_argument("--comodo", default=None, help="nome do cômodo (padrão: variável COMODO ou 'sala')")
    p.add_argument("--limite", type=float, default=None, help="segundos sem ninguém até abrir o evento")
    p.add_argument("--demo", action="store_true", help="limite de 10 s para demonstração")
    p.add_argument("--imgsz", type=int, default=640, help="tamanho de entrada do modelo")
    p.add_argument("--saida", default=None, help="grava o vídeo anotado neste arquivo .mp4")
    p.add_argument("--pre-treinado", action="store_true", help="usa pesos COCO (teste de integração antes do treino)")
    p.add_argument("--sem-janela", action="store_true", help="não abre janela (servidor ou processamento de vídeo)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    executar(args)


if __name__ == "__main__":
    main()
