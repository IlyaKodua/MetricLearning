from torch.utils.data import Dataset
import torch
import glob
import os
import numpy as np
from PIL import Image
import clip
from tqdm import tqdm
import pickle
from torch.utils.data import Sampler, SequentialSampler
from torch import optim
from sklearn.metrics import accuracy_score
 

def find_duplicates(lst):


  count_dict = {}
  for num in lst:
    count_dict[num] = count_dict.get(num, 0) + 1

  duplicates = [key for key, value in count_dict.items() if value > 1]
  return duplicates



class RetailDataset(Dataset):


    def __init__(self, root_dir, preprocess, emb_size = None):

        self.root_dir = root_dir
        self.files = sorted(glob.glob(root_dir + '/*.jpg', recursive=True))
        self.model_name = ""
        self.emb_size = emb_size
        self.preprocess = preprocess

        uniq_labels_list = []
        for file in self.files:
            uniq_labels_list.append(file.split("/")[-1].split("_")[0])

        self.uniq_labels_list = find_duplicates(uniq_labels_list)

        uniq_labels_list = sorted(list(uniq_labels_list))
        self.len_classes = len(uniq_labels_list)
        print("Number of classes: ", self.len_classes)
        self.labels_dict = {}

        for i, class_name in enumerate(uniq_labels_list):
            self.labels_dict[class_name] = i
    
    def calc_top3_by_cosine_similarity_matrix(self):
        embbeds1_normalized = self.embeddings_dict[0] / np.linalg.norm(self.embeddings_dict[0], axis=1, keepdims=True)
        embbeds2_normalized = self.embeddings_dict[1] / np.linalg.norm(self.embeddings_dict[1], axis=1, keepdims=True)

        # Calculate the cosine similarity matrix using matrix multiplication
        similarity_matrix = np.dot(embbeds1_normalized, embbeds2_normalized.T)

        self.top_3_indices = np.argpartition(-similarity_matrix, 3, axis=1)[:, :3]  # Descending order
    

    def calc_embeddings(self, model, device, train_indices):
        self.train_indices = train_indices
        self.embeddings_dict = {0 : np.zeros((len(self.train_indices), self.emb_size)),
                           1 : np.zeros((len(train_indices), self.emb_size))}

        with torch.no_grad():
            for i, id in tqdm(enumerate(self.train_indices), total=len(self.train_indices)):
                file = self.uniq_labels_list[id]
                path2file = self.root_dir + "/" + file
                file1 = path2file + "_1.jpg"
                file2 = path2file + "_2.jpg"
                image = self.preprocess(Image.open(file1)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                self.embeddings_dict[0][i] = image_features.detach().cpu().numpy()[0]

                file2 = path2file + "_2.jpg"
                image = self.preprocess(Image.open(file2)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                self.embeddings_dict[1][i] = image_features.detach().cpu().numpy()[0]

        with open('embed_dict.pickle', 'wb') as handle:
            pickle.dump(self.embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_embedding_dict(self):
        with open('embed_dict.pickle', 'rb') as handle:
            self.embeddings_dict = pickle.load(handle)

        
    def __len__(self):
        return len(self.uniq_labels_list)
    
    def __load_img(self, path):
            img = Image.open(path)
            if img.mode == "P":
                img = img.convert('RGBA')
            return img

    def __getitem__(self, idx):
        if type(idx) != int:
            anchor = self.__load_img(self.root_dir + "/" + self.uniq_labels_list[idx[0]] + "_1.jpg")
            positive =  self.__load_img(self.root_dir + "/" + self.uniq_labels_list[idx[0]] + "_2.jpg")
            negative =  self.__load_img(self.root_dir + "/" + self.uniq_labels_list[idx[1]] + "_2.jpg")
            return self.preprocess(anchor), self.preprocess(positive), self.preprocess(negative)
        else:
            anchor =  self.__load_img(self.root_dir + "/" + self.uniq_labels_list[idx] + "_1.jpg")
            positive =  self.__load_img(self.root_dir + "/" + self.uniq_labels_list[idx] + "_2.jpg")
            return self.preprocess(anchor), self.preprocess(positive)


class TriplerSampler(Sampler):
    """
    Sampler, который возвращает индексы данных в строгом порядке.

    Args:
        data_source (Dataset): Источник данных.
    """

    def __init__(self, data_source, train_ids):
        self.data_source = data_source
        self.indices = list(range(len(data_source)))
        self.num_samples = len(self.indices) 
        self.train_ids = train_ids
        self.top3_vector = self.data_source.dataset.top_3_indices

    def __iter__(self):
        for idx in self.indices:
            anchor_idx = idx
            
            # Положительный образец (следующий из того же класса)
            positive_idx = anchor_idx

            negative_idx =  self.top3_vector[anchor_idx][self.top3_vector[anchor_idx] != anchor_idx][0]

            if positive_idx == negative_idx:
                negative_idx = self.top3_vector[idx][1]
            
            yield self.train_ids[positive_idx], self.train_ids[negative_idx]


    def __len__(self):
        """
        Возвращает общее количество образцов.
        """
        return self.num_samples


def top_n(embbeds1, embbeds2, n):
    """Calculates the cosine similarity matrix between two sets of embeddings.

    Args:
        embbeds1: A NumPy array of shape (num_embeddings1, embedding_dim).
        embbeds2: A NumPy array of shape (num_embeddings2, embedding_dim).

    Returns:
        A NumPy array of shape (num_embeddings1, num_embeddings2) containing the cosine
        similarity scores between each pair of embeddings.
    """

    # Normalize the embeddings to unit length
    embbeds1_normalized = embbeds1 / np.linalg.norm(embbeds1, axis=1, keepdims=True)
    embbeds2_normalized = embbeds2 / np.linalg.norm(embbeds2, axis=1, keepdims=True)

    # Calculate the cosine similarity matrix using matrix multiplication
    similarity_matrix = np.dot(embbeds1_normalized, embbeds2_normalized.T)

    top_n_indices = np.argpartition(-similarity_matrix, 3, axis=1)[:,:n]  # Descending order
    return top_n_indices


def eval_script(model, dataloader):

    emb1_list = []
    emb2_list = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            emb_anchor = model(data[0].to(device)).detach().cpu().numpy()
            emb_positive = model(data[1].to(device)).detach().cpu().numpy()
            for i in range(emb_anchor.shape[0]):
                emb1_list.append(emb_anchor[i])
                emb2_list.append(emb_positive[i])
    top1 = top_n(np.array(emb1_list), np.array(emb2_list), 1)
    top5 = top_n(np.array(emb1_list), np.array(emb2_list), 5)
    cnt = 0
    for i in range(len(top5)):
        if i in top5[i]:
            cnt+=1
    return accuracy_score(top1, np.arange(len(emb1_list))), cnt/len(emb1_list)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel
    from pytorch_metric_learning.losses import SelfSupervisedLoss, TripletMarginLoss
    from pytorch_metric_learning.distances import CosineSimilarity
    from pytorch_metric_learning.reducers import ThresholdReducer
    from pytorch_metric_learning.regularizers import LpRegularizer
    loss_func = SelfSupervisedLoss(TripletMarginLoss(margin = 1,
                                                    distance = CosineSimilarity(), 
                                        reducer = ThresholdReducer(high=0.3), 
                                        embedding_regularizer = LpRegularizer()))

    epoches = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50x4", device=device)
    # model = torchvision.models.convnext_base(weights='DEFAULT')
    # model.classifier[2] = torch.nn.Identity()

    # processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')


    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

    model.classifier = torch.nn.Identity()

    model = model.to(device)
    # for param in model.parameters(): 
    #     param.requires_grad = False


    # for param in model.attnpool.parameters():
    #     param.requires_grad = True

    dataset = RetailDataset(root_dir = "dataset", preprocess = preprocess, emb_size = 1280)

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

    train_indices = train_set.indices

    # train_set.dataset.calc_embeddings(model, device, train_indices)
    train_set.dataset.load_embedding_dict()
    train_set.dataset.calc_top3_by_cosine_similarity_matrix()
    train_sampler = TriplerSampler(train_set, train_indices)

    train_dataloader = DataLoader(dataset, batch_size=24, sampler=train_sampler, num_workers=8)
    val_dataloader = DataLoader(val_set, batch_size=24, sampler= SequentialSampler(val_set), num_workers=8)
    test_dataloader = DataLoader(test_set, batch_size=24, sampler=SequentialSampler(test_set), num_workers=8)
    
    # model = model.visual.float()


    top1, top5 =  eval_script(model, val_dataloader)
    print("Start top1 and top5: ", top1,  " ", top5)
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    
    criterion = torch.nn.L1Loss()
    for epoch in range(epoches):
        for data in tqdm(train_dataloader):
            optimizer.zero_grad()
            anchor, positive, negative = data
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            # negative_emb = model(negative)
            loss = criterion(anchor_emb, positive_emb)
            loss.backward()
            optimizer.step()
        top1, top5 =  eval_script(model, val_dataloader)
        print("Epoch: ", epoch, "top1 and top5: ", top1,  " ", top5)



    # dataloader = DataLoader(dataset, batch_size=2)
    print("Finish")
