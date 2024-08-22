import os, torch, pickle
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

# Load pre-trained model
class PretrainModel:
    def __init__(self):
        self.model = torch.load("models/pretrained.pkl")
        self.model.eval()
        self.transformer =  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def get_feature(self, pil_image):
        image_tensor = self.transformer(pil_image)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            feature = self.model(image_tensor).squeeze().detach().cpu().numpy()
        return feature

# hàm đọc file chứa nhãn của các hình ảnh
# trả ra 1 dictionary với key là tên file, value là nhãn (1: có xe, 0: ko có xe) (nhãn y dùng để train)
def read_training_data_label():
    with open("data/image_label.pkl", "rb") as f:
        return pickle.load(f)

# hàm đọc các file ảnh trong thư mục data/images
# trả ra 1 dictionary với key là tên file, value là vector đặc trưng của file ảnh đó (vector x dùng để train)
def read_training_data():
    features_dict = {}
    folder_path = 'data/images'
    pretrained = PretrainModel()
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        pil_image = Image.open(img_path).convert("RGB")
        features_dict[filename] = pretrained.get_feature(pil_image)
    return features_dict

def train():
    data = read_training_data()
    labels = read_training_data_label()
    X = []
    y = []
    for filename, embs in data.items():
        X.append(embs)
        y.append(labels[filename])
    
    model = LogisticRegression()
    model.fit(X, y)
    pickle.dump(model, open('models/model.pickle', 'wb'))
    print('model trained')

# train()