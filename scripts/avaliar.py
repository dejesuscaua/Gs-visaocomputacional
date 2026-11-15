"""Avalia o modelo no conjunto de TESTE e grava as métricas para o relatório.

    python scripts/avaliar.py --dados caminho/para/data.yaml --modelo modelo/best.pt

Saídas em resultados/: metricas_teste.json, matriz de confusão e curva precisão x recall.
"""

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dados", required=True, help="data.yaml exportado do Roboflow")
    p.add_argument("--modelo", default="modelo/best.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", default="test", help="divisão avaliada (use sempre test no relatório)")
    args = p.parse_args()

    met = YOLO(args.modelo).val(data=args.dados, split=args.split, imgsz=args.imgsz, plots=True,
                                project="runs", name="avaliacao", exist_ok=True)
    nomes = met.names
    resumo = {
        "mAP50": round(float(met.box.map50), 3),
        "mAP50_95": round(float(met.box.map), 3),
        "precisao": round(float(met.box.mp), 3),
        "recall": round(float(met.box.mr), 3),
        "por_classe": {
            nomes[int(c)]: {
                "precisao": round(float(met.box.p[i]), 3),
                "recall": round(float(met.box.r[i]), 3),
                "mAP50": round(float(met.box.ap50[i]), 3),
                "mAP50_95": round(float(met.box.ap[i]), 3),
            }
            for i, c in enumerate(met.box.ap_class_index)
        },
    }
    saida = Path("resultados")
    saida.mkdir(exist_ok=True)
    (saida / "metricas_teste.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    pasta = Path(met.save_dir)
    for origem, destino in [("confusion_matrix_normalized.png", "matriz_confusao.png"), ("BoxPR_curve.png", "curva_pr.png")]:
        if (pasta / origem).exists():
            shutil.copy(pasta / origem, saida / destino)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
