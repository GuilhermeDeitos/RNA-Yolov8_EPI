# Importar as bibliotecas necessárias
from ultralytics import YOLO
import cv2


# Carregar o modelo treinado (substituir 'best.pt' pelo caminho correto do seu modelo salvo)
# Aqui assumimos que o melhor modelo treinado foi salvo em 'runs/train/exp/weights/best.pt'
trained_model_path = "./runs/detect/train7/weights/best.pt"
try:
    trained_model = YOLO(trained_model_path)
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

print(trained_model.names)


# # Iniciar a captura de vídeo da câmera
# cap = cv2.VideoCapture(0)  # 0 para a câmera padrão

# if not cap.isOpened():
#     print("Erro ao abrir a câmera.")
#     exit()

# while True:
#     # Ler um frame da câmera
#     ret, frame = cap.read()
#     if not ret:
#         print("Erro ao capturar o frame.")
#         break

#     # Realizar a inferência no frame atual usando o modelo treinado
#     results = trained_model(frame)
#     if len(results) == 0:
#         print("Nenhuma detecção foi feita neste frame.")

#     # Obter as caixas delimitadoras e desenhar no frame
#     for result in results:
#         print(result)
#         boxes = result.boxes
#         for box in boxes:
#             # Obter as coordenadas da caixa
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas do canto superior esquerdo e inferior direito
#             confidence = box.conf[0]  # Confiança da detecção
#             class_id = int(box.cls[0])  # ID da classe detectada
            
#             # Desenhar a caixa delimitadora na imagem
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Cor da caixa: azul
#             cv2.putText(frame, f'Class: {class_id}, Conf: {confidence:.2f}', (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     # Exibir o frame com as detecções
#     cv2.imshow("Detecções de EPIs - Pressione 'q' para sair", frame)

#     # Sair se a tecla 'q' for pressionada
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Liberar a captura e fechar as janelas
# cap.release()
# cv2.destroyAllWindows()