"""
Script educacional de detecção de objetos com OpenCV.

Modos disponíveis:
    1 - Detecção por cores (espaço HSV)
    2 - Detecção de formas geométricas
    3 - Detecção de faces e olhos (Haar Cascade)
    4 - Detecção de contornos gerais (Canny)

Controles:
    1-4  → Trocar modo de detecção
    S    → Salvar screenshot
    Q    → Sair

Uso:
    source .venv/bin/activate             # Ative o ambiente virtual primeiro
    python opencv_detect.py               # Usa webcam padrão (no macOS prioriza câmera física)
    python opencv_detect.py 1             # Usa webcam de índice 1
    python opencv_detect.py 2             # Exemplo: webcam de índice 2
    python opencv_detect.py video.mp4     # Usa arquivo de vídeo
"""

import cv2
import numpy as np
import sys
import os
from datetime import datetime


# =============================================================================
# Constantes de configuração
# =============================================================================

# Nomes dos modos para exibição na tela
MODOS = {
    1: "Deteccao por Cores (HSV)",
    2: "Deteccao de Formas Geometricas",
    3: "Deteccao de Faces e Olhos",
    4: "Deteccao de Contornos (Canny)",
}

# Faixa de cor para detecção HSV — padrão: verde
# Formato: [H_min, S_min, V_min], [H_max, S_max, V_max]
COR_HSV_MIN = np.array([35, 80, 40])
COR_HSV_MAX = np.array([85, 255, 255])

# Caminhos dos arquivos de Haar Cascade
CASCADE_FACE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CASCADE_EYE  = cv2.data.haarcascades + "haarcascade_eye.xml"


# =============================================================================
# Modo 1 — Detecção por cores (HSV)
# =============================================================================

def detectar_cores(frame: np.ndarray) -> np.ndarray:
    """
    Detecta objetos de uma cor específica usando o espaço de cores HSV.

    O espaço HSV (Hue, Saturation, Value) separa a informação de cor (Hue)
    da luminosidade (Value), tornando a segmentação por cor mais robusta
    a variações de iluminação do que o espaço RGB.

    Passos:
        1. Converter BGR → HSV
        2. Criar máscara binária com a faixa de cor desejada
        3. Aplicar erosão e dilatação (operações morfológicas) para remover ruído
        4. Encontrar contornos dos objetos detectados
        5. Desenhar retângulos ao redor dos objetos

    Parâmetros:
        frame: Imagem BGR capturada da câmera/vídeo.

    Retorna:
        Imagem anotada com os objetos detectados destacados.
    """
    resultado = frame.copy()

    # --- Passo 1: Converter para HSV ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Passo 2: Criar máscara binária (pixels brancos = cor encontrada) ---
    mascara = cv2.inRange(hsv, COR_HSV_MIN, COR_HSV_MAX)

    # --- Passo 3: Operações morfológicas para reduzir ruído ---
    # Erosão remove pixels brancos isolados (ruído pequeno)
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.erode(mascara, kernel, iterations=1)
    # Dilatação recupera o tamanho original dos objetos após a erosão
    mascara = cv2.dilate(mascara, kernel, iterations=2)

    # --- Passo 4: Encontrar contornos na máscara ---
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Passo 5: Desenhar retângulos ao redor dos objetos grandes ---
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area > 500:  # Ignorar objetos muito pequenos
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(resultado, f"Area: {int(area)}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Exibir a máscara no canto superior esquerdo (miniatura)
    h_frame, w_frame = frame.shape[:2]
    mini_h, mini_w = h_frame // 5, w_frame // 5
    mascara_rgb = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    mini = cv2.resize(mascara_rgb, (mini_w, mini_h))
    resultado[5:5 + mini_h, 5:5 + mini_w] = mini
    cv2.rectangle(resultado, (5, 5), (5 + mini_w, 5 + mini_h), (255, 255, 0), 1)
    cv2.putText(resultado, "Mascara HSV", (7, mini_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return resultado


# =============================================================================
# Modo 2 — Detecção de formas geométricas
# =============================================================================

def detectar_formas(frame: np.ndarray) -> np.ndarray:
    """
    Detecta e classifica formas geométricas (triângulo, quadrado, retângulo,
    pentágono, hexágono, círculo) usando aproximação de contornos.

    A função cv2.approxPolyDP reduz o número de vértices de um contorno,
    aproximando-o por um polígono mais simples. Contando os vértices do
    polígono resultante, identificamos a forma geométrica.

    Parâmetros:
        frame: Imagem BGR capturada da câmera/vídeo.

    Retorna:
        Imagem anotada com as formas geométricas identificadas.
    """
    resultado = frame.copy()

    # --- Pré-processamento: escala de cinza → blur → detecção de bordas ---
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blur gaussiano suaviza ruídos antes da detecção de bordas
    blur = cv2.GaussianBlur(cinza, (5, 5), 0)
    # Limiarização adaptativa funciona melhor com variações de iluminação
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Encontrar todos os contornos externos na imagem limiarizada
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 1000:  # Ignorar formas muito pequenas / ruído
            continue

        # Aproximar o contorno por um polígono
        perimetro = cv2.arcLength(cnt, True)
        # epsilon: tolerância da aproximação (2% do perímetro)
        epsilon = 0.02 * perimetro
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)

        # Classificar a forma pelo número de vértices
        if vertices == 3:
            nome, cor = "Triangulo", (0, 255, 255)
        elif vertices == 4:
            # Verificar se é quadrado ou retângulo pela razão dos lados
            x, y, w, h = cv2.boundingRect(approx)
            razao = w / float(h)
            if 0.9 <= razao <= 1.1:
                nome, cor = "Quadrado", (255, 0, 0)
            else:
                nome, cor = "Retangulo", (0, 0, 255)
        elif vertices == 5:
            nome, cor = "Pentagono", (255, 165, 0)
        elif vertices == 6:
            nome, cor = "Hexagono", (128, 0, 128)
        else:
            # Muitos vértices → provavelmente um círculo
            nome, cor = "Circulo", (0, 255, 0)

        # Desenhar o contorno e o rótulo
        cv2.drawContours(resultado, [approx], -1, cor, 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(resultado, nome, (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

    return resultado


# =============================================================================
# Modo 3 — Detecção de faces e olhos (Haar Cascade)
# =============================================================================

def detectar_faces(frame: np.ndarray,
                   detector_face: cv2.CascadeClassifier,
                   detector_olho: cv2.CascadeClassifier) -> np.ndarray:
    """
    Detecta faces humanas e olhos usando classificadores Haar Cascade.

    Haar Cascade é um método de detecção de objetos proposto por Viola e Jones
    (2001). Ele usa features retangulares (Haar-like features) treinadas com
    AdaBoost para criar um classificador em cascata muito eficiente.

    Parâmetros:
        frame:          Imagem BGR capturada da câmera/vídeo.
        detector_face:  Classificador carregado para detecção de faces.
        detector_olho:  Classificador carregado para detecção de olhos.

    Retorna:
        Imagem anotada com faces (azul) e olhos (verde) detectados.
    """
    resultado = frame.copy()

    # Haar Cascade opera em imagens em escala de cinza
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Equalização de histograma melhora o contraste e a detecção
    cinza = cv2.equalizeHist(cinza)

    # Detectar faces
    # scaleFactor: reduz a imagem a cada passo da pirâmide de escala
    # minNeighbors: quantos vizinhos confirmar para evitar falsos positivos
    faces = detector_face.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (fx, fy, fw, fh) in faces:
        # Desenhar retângulo ao redor da face
        cv2.rectangle(resultado, (fx, fy), (fx + fw, fy + fh), (255, 100, 0), 2)
        cv2.putText(resultado, "Face", (fx, fy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        # Buscar olhos somente dentro da região da face (ROI)
        roi_cinza = cinza[fy:fy + fh, fx:fx + fw]
        roi_color = resultado[fy:fy + fh, fx:fx + fw]

        olhos = detector_olho.detectMultiScale(
            roi_cinza,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(20, 20)
        )

        for (ex, ey, ew, eh) in olhos:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 220, 0), 2)
            cv2.putText(roi_color, "Olho", (ex, ey - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)

    # Mostrar contagem de faces detectadas
    cv2.putText(resultado, f"Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

    return resultado


# =============================================================================
# Modo 4 — Detecção de contornos gerais (Canny)
# =============================================================================

def detectar_contornos(frame: np.ndarray) -> np.ndarray:
    """
    Detecta e exibe todos os contornos da cena usando o algoritmo de Canny.

    O algoritmo de Canny (1986) é um detector de bordas em múltiplos estágios:
        1. Suavização com filtro Gaussiano (reduz ruído)
        2. Gradiente de intensidade (detecta variações abruptas)
        3. Supressão de não-máximos (afina as bordas)
        4. Limiarização por histerese com dois thresholds (threshold1, threshold2)

    Parâmetros:
        frame: Imagem BGR capturada da câmera/vídeo.

    Retorna:
        Imagem colorida com contornos desenhados sobre fundo preto/original.
    """
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blur antes do Canny reduz detecção de bordas espúrias por ruído
    blur = cv2.GaussianBlur(cinza, (5, 5), 0)

    # Canny: threshold1=50 (histerese baixa), threshold2=150 (histerese alta)
    bordas = cv2.Canny(blur, threshold1=50, threshold2=150)

    # Encontrar todos os contornos externos e internos
    contornos, hierarquia = cv2.findContours(bordas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Criar canvas preto para desenhar apenas os contornos
    canvas = np.zeros_like(frame)

    # Desenhar contornos com cores variadas por nível hierárquico
    for i, cnt in enumerate(contornos):
        if cv2.contourArea(cnt) < 100:
            continue
        # Cor baseada no índice do contorno (variedade visual)
        cor = (
            int((i * 37) % 256),
            int((i * 97) % 256),
            int((i * 53) % 256),
        )
        cv2.drawContours(canvas, [cnt], -1, cor, 1)

    # Misturar o canvas com o frame original usando alpha blending
    # resultado = alpha * frame + (1-alpha) * canvas
    alpha = 0.35
    resultado = cv2.addWeighted(frame, alpha, canvas, 1 - alpha, 0)

    # Exibir miniatura das bordas Canny no canto
    h_frame, w_frame = frame.shape[:2]
    mini_h, mini_w = h_frame // 5, w_frame // 5
    bordas_rgb = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    mini = cv2.resize(bordas_rgb, (mini_w, mini_h))
    resultado[5:5 + mini_h, 5:5 + mini_w] = mini
    cv2.rectangle(resultado, (5, 5), (5 + mini_w, 5 + mini_h), (0, 255, 255), 1)
    cv2.putText(resultado, "Bordas Canny", (7, mini_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    total = sum(1 for c in contornos if cv2.contourArea(c) >= 100)
    cv2.putText(resultado, f"Contornos: {total}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return resultado


# =============================================================================
# Painel de informações semi-transparente (HUD)
# =============================================================================

def desenhar_painel(frame: np.ndarray, modo: int, fps: float) -> np.ndarray:
    """
    Sobrepõe um painel semi-transparente com controles e modo atual.

    Usa Alpha Blending (cv2.addWeighted) para criar o efeito de transparência.
    Um retângulo sólido é desenhado em uma cópia do frame e depois mesclado
    com o original, criando a aparência translúcida.

    Parâmetros:
        frame: Imagem já processada pelo modo de detecção.
        modo:  Modo atual (1-4).
        fps:   Taxa de quadros estimada.

    Retorna:
        Imagem com o painel HUD sobreposto.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Fundo semi-transparente no canto inferior direito
    painel_w, painel_h = 290, 145
    px = w - painel_w - 10
    py = h - painel_h - 10
    cv2.rectangle(overlay, (px, py), (px + painel_w, py + painel_h), (20, 20, 20), -1)

    # Mesclar overlay com frame original (alpha = 0.55 → 55% opaco)
    resultado = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    # Textos do painel
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cor_titulo = (200, 200, 50)
    cor_texto  = (220, 220, 220)
    cor_ativo  = (50, 255, 50)

    cv2.putText(resultado, "CONTROLES", (px + 8, py + 20), fonte, 0.55, cor_titulo, 1)
    linhas = [
        ("[1] Cores HSV",          1),
        ("[2] Formas Geometricas", 2),
        ("[3] Faces + Olhos",      3),
        ("[4] Contornos Canny",    4),
        ("[S] Screenshot",         0),
        ("[Q] Sair",               0),
    ]
    for i, (texto, num) in enumerate(linhas):
        cor = cor_ativo if num == modo else cor_texto
        cv2.putText(resultado, texto, (px + 8, py + 40 + i * 17), fonte, 0.45, cor, 1)

    # FPS no canto superior direito
    cv2.putText(resultado, f"FPS: {fps:.1f}", (w - 100, 25), fonte, 0.6, (180, 180, 180), 1)

    # Nome do modo atual no topo central
    nome_modo = MODOS.get(modo, "")
    tw = cv2.getTextSize(nome_modo, fonte, 0.65, 2)[0][0]
    cv2.putText(resultado, nome_modo, ((w - tw) // 2, 30), fonte, 0.65, (50, 220, 255), 2)

    return resultado


# =============================================================================
# Função principal
# =============================================================================

def encontrar_webcam_padrao() -> int:
    """
    Escolhe um índice de webcam padrão.

    No macOS, tenta evitar câmera virtual do OBS (comum no índice 0)
    priorizando o índice 1. Se não funcionar, tenta outros índices.
    """
    if sys.platform != "darwin":
        return 0

    candidatos = [1, 0, 2, 3, 4]
    for indice in candidatos:
        cap_teste = cv2.VideoCapture(indice, cv2.CAP_AVFOUNDATION)
        if cap_teste.isOpened():
            ok, frame = cap_teste.read()
            cap_teste.release()
            if ok and frame is not None and frame.size > 0:
                return indice
        else:
            cap_teste.release()

    return 0

def main():
    """
    Ponto de entrada do script.

    Abre a câmera ou arquivo de vídeo, entra no loop principal de captura
    de frames e chama a função de detecção correspondente ao modo ativo.
    """
    # --- Carregar fonte de vídeo ---
    if len(sys.argv) > 1:
        entrada = sys.argv[1]

        # Se o argumento for inteiro, tratamos como índice da webcam
        try:
            fonte = int(entrada)
            print(f"[INFO] Usando webcam de índice {fonte}")
        except ValueError:
            fonte = entrada
            if not os.path.exists(fonte):
                print(f"[ERRO] Fonte inválida: {fonte}")
                print("       Use o índice da câmera (ex.: 0, 1) ou um caminho de vídeo válido.")
                sys.exit(1)
            print(f"[INFO] Usando arquivo de vídeo: {fonte}")
    else:
        fonte = encontrar_webcam_padrao()
        print(f"[INFO] Usando webcam padrão (índice {fonte})")

    if isinstance(fonte, int) and sys.platform == "darwin":
        cap = cv2.VideoCapture(fonte, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(fonte)

    if not cap.isOpened():
        print("[ERRO] Não foi possível abrir a fonte de vídeo.")
        print("       Verifique se a webcam está conectada ou se o arquivo existe.")
        sys.exit(1)

    # --- Carregar classificadores Haar Cascade ---
    if not os.path.exists(CASCADE_FACE):
        print("[AVISO] Arquivo de cascade de face não encontrado.")
        print(f"        Esperado em: {CASCADE_FACE}")

    detector_face = cv2.CascadeClassifier(CASCADE_FACE)
    detector_olho  = cv2.CascadeClassifier(CASCADE_EYE)

    if detector_face.empty():
        print("[AVISO] Cascade de face não carregado — modo 3 pode não funcionar.")

    # --- Configurações iniciais ---
    modo_atual   = 1
    screenshot_n = 0
    prev_tick    = cv2.getTickCount()

    print("\n" + "="*50)
    print("  Detecção de Objetos com OpenCV — Educacional")
    print("="*50)
    print("  Teclas: 1-4 (modo)  |  S (screenshot)  |  Q (sair)")
    print("="*50 + "\n")

    # --- Loop principal ---
    while True:
        ret, frame = cap.read()

        # Fim do vídeo: reiniciar do início
        if not ret:
            if isinstance(fonte, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[AVISO] Não foi possível capturar frame da webcam.")
                break

        # Espelhar horizontalmente (efeito espelho — mais intuitivo com webcam)
        if fonte == 0:
            frame = cv2.flip(frame, 1)

        # --- Calcular FPS ---
        tick_atual = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (tick_atual - prev_tick)
        prev_tick = tick_atual

        # --- Processar frame conforme o modo ativo ---
        try:
            if modo_atual == 1:
                frame_proc = detectar_cores(frame)
            elif modo_atual == 2:
                frame_proc = detectar_formas(frame)
            elif modo_atual == 3:
                frame_proc = detectar_faces(frame, detector_face, detector_olho)
            else:
                frame_proc = detectar_contornos(frame)
        except Exception as e:
            # Em caso de erro no processamento, exibir frame original com aviso
            frame_proc = frame.copy()
            cv2.putText(frame_proc, f"Erro: {str(e)[:60]}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # --- Desenhar painel HUD ---
        frame_final = desenhar_painel(frame_proc, modo_atual, fps)

        # --- Exibir resultado ---
        cv2.imshow("OpenCV - Deteccao de Objetos", frame_final)

        # --- Processar teclas (aguardar 1ms) ---
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('q') or tecla == ord('Q'):
            print("[INFO] Encerrando...")
            break
        elif tecla == ord('1'):
            modo_atual = 1
            print(f"[INFO] Modo: {MODOS[1]}")
        elif tecla == ord('2'):
            modo_atual = 2
            print(f"[INFO] Modo: {MODOS[2]}")
        elif tecla == ord('3'):
            modo_atual = 3
            print(f"[INFO] Modo: {MODOS[3]}")
        elif tecla == ord('4'):
            modo_atual = 4
            print(f"[INFO] Modo: {MODOS[4]}")
        elif tecla == ord('s') or tecla == ord('S'):
            # Salvar screenshot com timestamp
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"screenshot_{ts}_{screenshot_n:03d}.png"
            cv2.imwrite(nome_arquivo, frame_final)
            screenshot_n += 1
            print(f"[INFO] Screenshot salvo: {nome_arquivo}")

    # --- Limpeza ---
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Recursos liberados. Até mais!")


if __name__ == "__main__":
    main()
