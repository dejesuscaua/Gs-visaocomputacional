# Dataset SeguWatts Vision

A entrega usa um dataset público do Roboflow Universe, baixado pelo notebook (seção abaixo). As outras seções são o nosso plano para a versão 2: um dataset próprio, com fotos das casas da equipe, mais fiel ao ambiente real. As imagens **não** ficam neste repositório (tamanho e privacidade).

## Classes

| ID | Classe | Anotar quando | Não anotar |
|---|---|---|---|
| 0 | `lampada_acesa` | lâmpada, luminária, abajur ou plafon claramente emitindo luz | reflexo da lâmpada em vidro ou espelho |
| 1 | `lampada_apagada` | o mesmo objeto, desligado | luz do sol entrando pela janela |
| 2 | `tela_ligada` | TV ou monitor com imagem, inclusive descanso de tela | celular na mão (fora do escopo) |
| 3 | `tela_desligada` | TV ou monitor preto | quadro ou espelho escuro |
| 4 | `pessoa` | pessoa inteira ou parcial (mais de 1/3 do corpo visível) | pessoa em foto, quadro ou na própria TV |

A ordem dos IDs precisa ser a mesma no Roboflow. Exporte com estas classes e nesta ordem.

## Opção usada na entrega: dataset público

O treino usa um dataset público já anotado do Roboflow Universe, baixado pelo notebook:

| Item | Valor |
|---|---|
| Dataset | [Light Detection · infocomm-thingy](https://universe.roboflow.com/infocomm-thingy/light-detection-gecjq) |
| Classes originais | `Lamp On`, `Lamp Off`, `Light Bulb On`, `Light Bulb Off` |
| Tradução | `*On` → `lampada_acesa`; `*Off` → `lampada_apagada` (`normalizar_classe`) |
| Versão | 3 (exportada em 29/08/2024) |
| Imagens | 695 (treino 603, validação 58, teste 34) |
| Licença | CC BY 4.0 (uso permitido com atribuição) |

Pessoas vêm da YOLO pré-treinada no COCO; por isso o dataset não precisa da classe `pessoa`. As fotos próprias abaixo continuam válidas para melhorar o modelo nas casas reais (versão 2).

## Protocolo de coleta (cerca de 100 fotos por integrante)

Fotografe com o celular na horizontal, do ponto onde a câmera ficaria (canto alto do cômodo, como uma câmera de segurança). Cubra todas as combinações:

| Situação | Fotos |
|---|---|
| Luz acesa, ninguém | 15 |
| Luz acesa, com pessoa | 15 |
| Luz apagada, ninguém (dia) | 10 |
| TV ligada, ninguém | 15 |
| TV ligada, com pessoa | 15 |
| TV desligada, com e sem pessoa | 10 |
| Luz e TV ligadas ao mesmo tempo | 10 |
| Casos difíceis: contraluz, noite com só a TV iluminando, pessoa de costas ou parcialmente fora do quadro | 10 |

Variações obrigatórias: pelo menos 2 cômodos por casa, dia e noite, 3 ângulos diferentes. Variedade de casas vale mais do que quantidade de fotos parecidas.

**Consentimento e LGPD:** fotografe só pessoas que concordaram (integrantes e familiares que autorizaram). Não publique as imagens. Evite documentos, telas com dados pessoais e janelas com vista para vizinhos. Ao exportar, não inclua metadados de localização (o Roboflow remove o EXIF).

## Anotação no Roboflow

1. Crie um projeto *Object Detection* com as 5 classes na ordem da tabela.
2. Faça o upload das fotos e anote com caixas justas ao objeto. Para lâmpadas, anote o conjunto da luminária, não só o ponto de luz.
3. Revise: cada integrante revisa as fotos de outro, porque o erro de rótulo é a principal causa de mAP baixo.
4. *Generate → Version*:
   - Pré-processamento: auto-orient e resize para 640×640 (fit, com bordas).
   - Aumento (só no treino, 3×): flip horizontal, brilho ±25%, exposição ±15%, rotação ±10°, blur até 1 px.
   - **Não** use aumentos de cor (hue e saturação): eles mudam a aparência de "aceso" e "apagado".
5. Exporte em *YOLOv8* e use a chave da API no notebook pelos *Secrets* do Colab.

## Contagem por classe

Gerada pela célula 3 do notebook e salva em `resultados/contagem_classes.csv`. Copie a tabela para o relatório e verifique o equilíbrio: nenhuma classe deve ter menos que metade das instâncias da maior.
