from torchvision.models import resnet50, ResNet50_Weights

from dataset import GlaucomaHarvardDatasetOld

weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
train_data = GlaucomaHarvardDatasetOld("train")
example_image = train_data.__getitem__(0)['image']

model = resnet50(weights=weights)
model.eval()

predictions = model(example_image).squeeze(0).softmax(0)
class_id = predictions.argmax().item()

print(f'Class ID: {class_id}, category: {weights.meta["categories"][class_id]}')
print(f' with confidence: {predictions[class_id].item()}')
