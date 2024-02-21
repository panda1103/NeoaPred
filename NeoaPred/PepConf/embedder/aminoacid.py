import torch.nn as nn

class AminoacidEmbedding(nn.Embedding):
    def __init__(self, aminoacid_size, dim):
        super().__init__(num_embeddings=aminoacid_size, embedding_dim=dim)
