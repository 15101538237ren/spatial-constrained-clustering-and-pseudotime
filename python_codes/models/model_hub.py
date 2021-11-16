# -*- coding: utf-8 -*-
import torch
from torch_geometric.nn import DeepGraphInfomax
from .dgi import DGI_Encoder

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def get_model(args, n_genes):
    model = DeepGraphInfomax(
                    hidden_channels=args.z_dim, encoder=DGI_Encoder(n_genes, args.z_dim),
                    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
                    corruption=corruption).to(args.device)
    return model