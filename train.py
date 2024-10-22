from ultralytics import YOLO
import matplotlib.pyplot as plt

class TrainModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.results = None
    
    def train(self, data, epochs):
        # Continuar o treinamento se 'resume' for True
        results = self.model.train(data=data, epochs=epochs)
        return results
    
    def val(self):
        self.results = self.model.val()
        results = self.results
        return results
    
    def plot_results(self):
        print("Results: ", self.results)
        self.results.plot()
        plt.show()
    
    def export(self, format):
        success = self.model.export(format=format)
        return success


# Definir o caminho para o modelo YOLO pré-treinado
model_path = "/results_yolov8n_100e/kaggle/working/yolov8n.pt"

#Inicializar o objeto
train_model = TrainModel(model_path)

# Treinar o modelo usando o conjunto de dados 'coco8.yaml' por 25 épocas
results = train_model.train(data="coco8.yaml", epochs=200)

# Avaliar o desempenho do modelo no conjunto de validação
results = train_model.val()

train_model.plot_results()

# Exportar o modelo para o formato ONNX
success = train_model.export(format="onnx")