import cv2
import numpy as np

# Caminho para a pasta de modelos
model_path = "modelos/"

# Carregar o modelo de detecção de objetos pré-treinado (SSD)
config_file = model_path + "ssdlite_mobilenet_v3_large_320x320_coco.config"
model_weights = model_path + "frozen_inference_graph.pb"

net = cv2.dnn.readNet(model_weights, config_file)

# Carregar as classes
class_names_file = model_path + "coco.names"
with open(class_names_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Obter os nomes de todas as camadas do modelo
layer_names = net.getLayerNames()

# Obter os índices das camadas de saída
output_layers_indices = net.getUnconnectedOutLayers()

# Converter os índices para nomes de camadas
output_layers = [layer_names[i[0] - 1] for i in output_layers_indices]

# Capturar o vídeo com resolução reduzida
cap = cv2.VideoCapture("video.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Definir largura do frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Definir altura do frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detectar objetos no frame
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Processar as detecções
    for detection in detections[0, 0, :, :]:
        confidence = detection[2]
        if confidence > 0.5:
            class_id = int(detection[1])
            class_name = classes[class_id]
            box = detection[3:7] * np.array([width, height, width, height])
            x, y, w, h = box.astype('int')

            # Desenhar a bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Identificar o objeto detectado
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exibir o frame com as detecções
    cv2.imshow('Detecção de Objetos', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar o objeto de captura e fechar as janelas
cap.release()
cv2.destroyAllWindows()
