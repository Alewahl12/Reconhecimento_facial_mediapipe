#coleta.py
import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# Nome da pessoa
nome_pessoa = input("Digite o nome da pessoa: ")

# Criar um diretório para salvar os dados se não existir
if not os.path.exists('dados_landmarks'):
    os.makedirs('dados_landmarks')

# Inicializa a captura de vídeo usando a câmera do notebook
cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)

# Nome do arquivo CSV
arquivo_csv = f'dados_landmarks/{nome_pessoa}.csv'

# Função para normalizar as landmarks
def normalizar_landmarks(landmarks):
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    z_coords = [lm.z for lm in landmarks]

    x_mean, y_mean, z_mean = np.mean(x_coords), np.mean(y_coords), np.mean(z_coords)
    x_std, y_std, z_std = np.std(x_coords), np.std(y_coords), np.std(z_coords)

    normalized_landmarks = [(lm.x - x_mean) / x_std for lm in landmarks] + \
                           [(lm.y - y_mean) / y_std for lm in landmarks] + \
                           [(lm.z - z_mean) / z_std for lm in landmarks]
    return normalized_landmarks

# Função para aumentar os dados
def augment_landmarks(landmarks, num_augmentations=5, noise_level=0.01):
    augmented_landmarks = []
    for _ in range(num_augmentations):
        noise = np.random.normal(0, noise_level, len(landmarks))
        augmented_landmarks.append(landmarks + noise)
    return augmented_landmarks

# Abrir o arquivo CSV para escrita
with open(arquivo_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([f'x{i}' for i in range(468)] + [f'y{i}' for i in range(468)] + [f'z{i}' for i in range(468)] + ['label'])

    contador = 0
    qtdMax_dados = 5000  # Alvo de 10.000 amostras

    while contador < qtdMax_dados:
        success, img = cap.read()
        if not success:
            print("Erro ao ler o quadro da câmera.")
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                normalized_landmarks = normalizar_landmarks(faceLms.landmark)
                augmented_data = augment_landmarks(np.array(normalized_landmarks))
                for aug_landmarks in augmented_data:
                    writer.writerow(aug_landmarks.tolist() + [nome_pessoa])
                    contador += 1
                    print(f"Landmarks capturadas e salvas para {nome_pessoa}. Total: {contador}/{qtdMax_dados}")
                
                # Desenhar landmarks e traços do rosto
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,
                                      mpDraw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),  # Pontos vermelhos
                                      mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))  # Traços verdes
                
        cv2.imshow("Coleta de Dados", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
