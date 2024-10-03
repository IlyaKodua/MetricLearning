import pickle
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import accuracy_score
import os
import shutil


if os.path.exists("out"):
   shutil.rmtree("out")
os.mkdir("out")


def cosine_similarity_matrix(embbeds1, embbeds2):
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

    top_3_indices = np.argpartition(-similarity_matrix, 3, axis=1)[:, :3]  # Descending order
    return top_3_indices



def find_duplicates(lst):


  count_dict = {}
  for num in lst:
    count_dict[num] = count_dict.get(num, 0) + 1

  duplicates = [key for key, value in count_dict.items() if value > 1]
  return duplicates


with open(r"embed_dict.pickle", "rb") as input_file:
    data = pickle.load(input_file)


embed_dict = {}
uniq_labels_list = []
for file in data.keys():
    uniq_labels_list.append(file.split("_")[0])



uniq_labels_list = find_duplicates(uniq_labels_list)

embbeds1 = np.zeros((len(uniq_labels_list), 768), dtype=np.float32)
embbeds2 = np.zeros((len(uniq_labels_list), 768), dtype=np.float32)
files_list = []
for i, file in enumerate(uniq_labels_list):
    file1 = file + "_1.jpg"
    file2 = file + "_2.jpg"
    embbeds1[i, :] = data[file1]
    embbeds2[i, :] = data[file2]
    files_list.append(file)


similarity_matrix = cosine_similarity_matrix(embbeds1, embbeds2)
top_3_indices = np.argmax(similarity_matrix, axis=1)

cnt = 0
# for i in range(len(best_id)):
#     if best_id[i] != i:
#        namei = files_list[i].split("/")[-1]
#        namej = files_list[best_id[i]].split("/")[-1]
#        shutil.copy(files_list[i] + "_1.jpg", "out/" + str(cnt) + "_1_" + namei + ".jpg")
#        shutil.copy(files_list[i] + "_2.jpg", "out/" + str(cnt) + "_2_" + namej + ".jpg")
#        cnt += 1
# print(accuracy_score(best_id, np.arange(len(uniq_labels_list))))



# best_id = np.zeros((len(uniq_labels_list)), dtype=np.int32)
# for i in range(len(uniq_labels_list)):
#     dist_vector = np.zeros((len(uniq_labels_list)), dtype=np.float32)
#     for j in range(len(uniq_labels_list)):
#        dist_vector[j] = np.sum(embbeds1[i]*embbeds2[j]) / (np.sqrt(np.sum(embbeds1[i]**2)) * np.sqrt(np.sum(embbeds1[j]**2)))
#     best_id[i] = np.argmax(dist_vector)
# print(1)