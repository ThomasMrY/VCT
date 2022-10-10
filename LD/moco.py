import torch
import torch.nn as nn
# from stylegan2.VAE_deep import GANbaseline

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=64, K=4096, T=1, load=False, dataset="shapes3d"):
        """
        dim: feature dimension (default: 64)
        K: queue size; number of negative keys (default: 4096)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.T = T

#         # create the encoders
#         VAE = base_encoder(nc = 3, z_dim = 64, group=False)
#         if load:
#             if dataset == "shapes3d":
#                 VAE.load_state_dict(torch.load("./generators/3dshapes/model_deep")['model_states']['net'])
                
#         self.encoder = VAE.encoder()
        
#         if load:
#             for param in self.encoder.parameters():
#                     param.requires_grad = False  # not update by gradient
#             self.encoder.eval()

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        
        queue_label = torch.randint(0, 2, [K])
        self.register_buffer("queue_label", queue_label)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, label):
        # gather keys before updating queue
#         keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_label[ptr:ptr + batch_size] = label
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, q, key, label):
        """
        Output:
            logits, targets
        """
        queue = self.queue.clone().detach().permute(1,0)
        queue_label = self.queue_label.clone().detach()
        
        select_queue = queue[queue_label != label].permute(1,0)
        
        K = select_queue.shape[1]
        N = q.shape[0]
        
        # compute logits
        # Einstein sum is more intuitive
        l_pos = torch.einsum('nc,ck->nk', [q, key.permute(1,0)])
        l_neg = torch.einsum('nc,ck->nk', [q, select_queue])

        # logits: Nx(N+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T
        
        labels = torch.zeros(N + K, dtype=torch.long).cuda()
        labels[range(N)] = 1
        labels = labels.unsqueeze(0).repeat(N, 1).float()

        # dequeue and enqueue
        self._dequeue_and_enqueue(key, label)

        return logits, labels


# # utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output