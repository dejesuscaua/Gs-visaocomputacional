import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from seguwatts_vision.eventos import SaidaEventos
from seguwatts_vision.regra import Evento


def evento_exemplo():
    ev = Evento("sala", ["lampada_acesa", "tela_ligada"], inicio=1_000.0, aberto_em=1_120.0)
    ev.encerrado_em = 1_300.0
    return ev


def test_grava_csv_sem_webhook(tmp_path):
    arquivo = tmp_path / "eventos.csv"
    saida = SaidaEventos(str(arquivo), tarifa_rs_kwh=0.85)
    dados = saida.publicar(evento_exemplo())
    assert dados["enviado"] is False
    linhas = list(csv.DictReader(arquivo.open(encoding="utf-8")))
    assert len(linhas) == 1
    assert linhas[0]["aparelhos"] == "lampada_acesa;tela_ligada"
    assert linhas[0]["duracao_s"] == "300.0"
    # 102 W por 300 s = 0,0085 kWh
    assert float(linhas[0]["energia_estimada_kwh"]) == 0.0085


def test_webhook_recebe_json_sem_imagem(tmp_path):
    recebido = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            tamanho = int(self.headers["Content-Length"])
            recebido.update(json.loads(self.rfile.read(tamanho)))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    servidor = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=servidor.handle_request, daemon=True).start()
    url = f"http://localhost:{servidor.server_port}/webhook"

    saida = SaidaEventos(str(tmp_path / "e.csv"), 0.85, webhook_url=url)
    dados = saida.publicar(evento_exemplo())
    servidor.server_close()

    assert dados["enviado"] is True
    assert recebido["comodo"] == "sala"
    assert recebido["tipo"] == "encerrado"
    assert not any(k in recebido for k in ("imagem", "frame", "foto"))


def test_webhook_fora_do_ar_nao_derruba(tmp_path):
    saida = SaidaEventos(str(tmp_path / "e.csv"), 0.85, webhook_url="http://localhost:9/nada", timeout_s=0.5)
    assert saida.publicar(evento_exemplo())["enviado"] is False


def test_webhook_sem_https_e_ignorado(tmp_path):
    saida = SaidaEventos(str(tmp_path / "e.csv"), 0.85, webhook_url="http://exemplo.invalid/hook")
    assert saida.publicar(evento_exemplo())["enviado"] is False
