import pandas as pd
import time
from os import path
from os import listdir
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import h5py

images_dir_path = path.join("images")

device = "cpu"

resnet = models.resnet50(pretrained=True).to(device)
resnet.eval()  # Set to evaluation mode
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Removes the last FC layer
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

items_df = pd.read_csv("new_items.csv")

def open_image(image_path):
  image = Image.open(image_path).convert("RGB")
  image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
  return image



def append_vectors(new_vectors, filename="image2vec.h5"):
    with h5py.File(filename, "a") as f:  
        dataset = f["embeddings"]
        old_size, vector_dim = dataset.shape
        new_size = old_size + new_vectors.shape[0]

        dataset.resize((new_size, vector_dim))
        
        dataset[old_size:new_size, :] = new_vectors


for i in range(0, len(items_df), 2000):
  j,k = i, i+2000
  print("read images")
  t1= time.time()
  images_name = items_df[j:k]['ID']
  image = torch.cat([open_image(path.join(images_dir_path, str(image_name)+'.jpg')) for image_name in images_name]).to(device)
  t2 = time.time()
  print(f'reading time: {t2-t1}')
  print(image.shape)

  print("embedding")
  with torch.no_grad():
      features = resnet(image)
  t1 = time.time()
  print(f'resnet time: {t1-t2}')

  image_embedding = features.view(features.size(0), -1)
  print("Image Embedding Shape:", image_embedding.shape)  # (2000, 2048)

  print("storing vectors")
  if(i==0):
      with h5py.File("image2vec.h5", "w") as f:
        f.create_dataset("embeddings", data=image_embedding)
  else:
      append_vectors(image_embedding)    
  print(f'vectors of elementes:[{j},{k}] has been stored.')
  


