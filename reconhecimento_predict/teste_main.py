#teste_main.py
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')


# Carregar o modelo treinado
with open('modelo.pkl', 'rb') as f:
    model = pickle.load(f)

# Inicializa a captura de vídeo usando a câmera do notebook
cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)

def normalize_landmarks(landmarks):
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    z_coords = [lm.z for lm in landmarks]

    x_mean, y_mean, z_mean = np.mean(x_coords), np.mean(y_coords), np.mean(z_coords)
    x_std, y_std, z_std = np.std(x_coords), np.std(y_coords), np.std(z_coords)

    normalized_landmarks = [(lm.x - x_mean) / x_std for lm in landmarks] + \
                           [(lm.y - y_mean) / y_std for lm in landmarks] + \
                           [(lm.z - z_mean) / z_std for lm in landmarks]
    return normalized_landmarks

while True:
    success, img = cap.read()
    if not success:
        print("Erro ao ler o quadro da câmera.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            normalized_landmarks = normalize_landmarks(faceLms.landmark)
            landmarks = np.array(normalized_landmarks).reshape(1, -1)
            
            # Fazer a previsão
            prediction = model.predict(landmarks)
            proba = model.predict_proba(landmarks)
            confianca = np.max(proba)

            if confianca > 0.80:  # Ajustar o limiar de confiança conforme necessário
                name = prediction[0]
                label = f'{name} {confianca:.2f}'
            else:
                label = "Desconhecido"

            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION)
            cv2.putText(img, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento Facial", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
