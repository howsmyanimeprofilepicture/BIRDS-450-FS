from batch_loader import Batch, BatchLoader
import torch
import torch.nn.functional as F

class ProtoNetTrainer:
    def __init__(self,
                 batch_sampler: BatchLoader,
                 encoder: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 ) -> None:
        self.batch_sampler = batch_sampler
        self.encoder = encoder
        self.optimizer = optimizer

    def train(self, epoch=5000):
        self.encoder.train()
        rec_losses = []
        losses = 0
        for i in range(epoch):
            batch, id2cls = self.batch_sampler.sample()
            loss, logits, = self.get_loss(batch)
            temp_loss = loss.item()
            losses += temp_loss
            if i % 100 == 99:
                rec_losses.append(losses/100)
                print(f"current loss: {rec_losses[-1]}")
                losses = 0

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

    def create_prototypes(self,
                          X_spt: torch.Tensor,
                          y_spt: torch.LongTensor,
                          ) -> torch.Tensor:
        N, c, h, w = X_spt.size()
        assert y_spt.size() == (N,)
        X_emb_spt: torch.Tensor = self.encoder(X_spt)
        _, emb_size = X_emb_spt.size()
        assert X_emb_spt.size() == (N, emb_size)

        prototypes = torch.stack([X_emb_spt[y_spt == i].mean(dim=0)
                                  for i in y_spt.detach().unique()])
        assert prototypes.size(1) == emb_size
        assert prototypes.ndim == 2

        return prototypes

    def get_loss(self,
                 batch: Batch):

        X_spt, X_qry, y_spt, y_qry = batch
        prototypes = self.create_prototypes(X_spt,  y_spt)

        # create l2 dist, logits, and loss
        X_emb_qry: torch.Tensor = self.encoder(X_qry)
        query_N, emb_size = X_emb_qry.size()

        num_way, p_emb_size = prototypes.size()
        assert query_N == (self.batch_sampler.num_qry 
                           * self.batch_sampler.num_way)
        assert emb_size == p_emb_size
        assert self.batch_sampler.num_way == num_way
        prototypes = prototypes.unsqueeze(0)
        prototypes = prototypes.expand(query_N, num_way,
                                       emb_size)
        prototypes = prototypes.reshape(query_N * num_way,
                                        emb_size)

        X_emb_qry = X_emb_qry.unsqueeze(1).expand(query_N, num_way, 256)
        X_emb_qry = X_emb_qry.reshape(query_N*num_way, emb_size)
        l2dist = torch.sum((prototypes - X_emb_qry)**2, 
                           dim=-1
                           ).reshape(num_way, query_N)
        logits = -l2dist
        loss = F.cross_entropy(logits, y_qry)
        logits = logits.cpu().detach()
        # preds = logits.argmax(dim=-1)
        return loss, logits, # preds.eq(y_qry).sum().item()
