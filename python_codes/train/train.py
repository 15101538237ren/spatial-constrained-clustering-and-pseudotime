# -*- coding: utf-8 -*-
import random
import torch
import numpy as np
import torch.nn.functional as F

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def train(args, model, expr, sp_graph, sp_dists, random_seed = 42):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    tail = '+ SP' if args.spatial else ''
    print(f'Training {args.arch} {tail}')

    model = model.to(args.device)
    expr = expr.to(args.device)
    sp_dists = sp_dists.to(args.device)
    edge_list = sparse_mx_to_torch_edge_list(sp_graph).to(args.device)

    model.train()
    min_loss = np.inf
    patience = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = 0.0
        torch.set_grad_enabled(True)
        optimizer.zero_grad()
        z, neg_z, summary = model(expr, edge_list)
        loss = model.loss(z, neg_z, summary)

        if args.spatial:
            n, in_dim = expr.shape
            z_dists = F.normalize(torch.cdist(z, z, p=2))
            penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n * n).to(args.device)
            loss = loss + args.penalty_scaler * penalty_1

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if train_loss > min_loss:
            patience += 1
        else:
            patience = 0
            min_loss = train_loss
        if epoch % 10 == 1:
            print("Epoch %d/%d" % (epoch + 1, args.epochs))
            print("Loss:" + str(train_loss))
        if patience > args.patience and epoch > args.min_stop:
            break
    z, _, _ = model(expr, edge_list)
    return z.cpu().detach().numpy()