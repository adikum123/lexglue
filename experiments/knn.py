import torch


# class for the knn algorithm
class kNN:
    def __init__(self, train_set_logits, config=None):
        self.config = config
        # shape (10, 10, 8) batch of size 10 padded so all seq have 10 words
        # each word encoded as a 8 dimensional vector
        self.data = torch.stack(train_set_logits)

    def get_k_nearest_neighbors(self, query_vector, k):
        distances = torch.cdist(self.data.view(self.data.shape[0], -1), query_vector.view(1, -1), p=2)
        _, nearest_indices = torch.topk(distances.squeeze(), k=k, largest=False)
        return self.data[nearest_indices]

    # for making predictions we pass a predict_set
    def predict(self, predict_set):
        pass

    def sltc(self):
        pass

    def mltc(self):
        pass

    def qa(self):
        pass

    def ner(self):
        pass
