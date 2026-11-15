# Roteiro do vídeo de demonstração (cerca de 3 min)

Grave a tela com o OBS ou outro gravador e a webcam apontada para um cômodo com luz e TV. Rode antes:

```bash
python -m seguwatts_vision.detector --demo --comodo sala --saida resultados/demo.mp4
```

O `--demo` reduz o tempo-limite para 10 s, para a cena caber no vídeo. A tecla `d` alterna entre 10 s e o limite normal.

| Tempo | Cena | Fala sugerida |
|---|---|---|
| 0:00–0:20 | Vitrine do SeguWatts no navegador | "A conta de luz mostra só o total. Um desperdício comum nenhum medidor enxerga: o aparelho ligado sem ninguém usando." |
| 0:20–0:45 | Fotos do dataset no Roboflow, contagem por classe | "Fotografamos as nossas casas: [N] imagens, 5 classes, anotadas e revisadas pela equipe." |
| 0:45–1:15 | Notebook: curvas de treino, matriz de confusão, tabela de métricas | "YOLO11 nano treinado a partir do COCO. No teste: mAP@50 de [X]. O COCO sabe o que é uma TV; o nosso modelo sabe se ela está ligada." |
| 1:15–2:15 | **Ao vivo:** (1) pessoa na sala com luz e TV: painel verde "ocupado"; (2) a pessoa sai: barra de contagem; (3) depois de 10 s: faixa magenta "Possível desperdício"; (4) a pessoa volta: o alerta some | Narre cada mudança de estado. Mostre no canto o FPS. |
| 2:15–2:35 | Execução no n8n recebendo o webhook e o e-mail ao morador; o `resultados/eventos.csv` aberto | "O evento leva só cômodo, aparelhos, horário e custo estimado. Nenhuma imagem sai da casa." |
| 2:35–3:00 | Slide de limitações e próximos passos | "Limites: dataset de cinco casas e potência estimada. Próximo passo: rodar na ESP32-S3 do EcoBot." |

Checklist antes de gravar:

- [ ] Modelo `modelo/best.pt` no lugar e testado no mesmo cômodo da gravação.
- [ ] Webhook do n8n configurado em `SEGUWATTS_WEBHOOK_URL`, com o fluxo ativo.
- [ ] Iluminação parecida com a do dataset (o modelo foi treinado com dia e noite).
- [ ] Nenhum documento, tela com dado pessoal ou pessoa que não autorizou no enquadramento.
