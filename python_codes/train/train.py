# -*- coding: utf-8 -*-
import random
import torch
import numpy as np
from python_codes.models.model_hub import get_model

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def train(args, adata_filtered, sp_graph, subsetting=True, random_seed = 42, random_subset_size=int(2e6), ccc=True):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    model = get_model(args, adata_filtered.shape[1])
    tail = '+ SP' if args.spatial else ''
    print(f'Training {args.arch} {tail}')

    model = model.to(args.device)
    expr = adata_filtered.X.todense() if type(adata_filtered.X).__module__ != np.__name__ else adata_filtered.X
    expr = torch.tensor(expr).float().to(args.device)

    edge_list = sparse_mx_to_torch_edge_list(sp_graph).to(args.device)
    model.train()
    min_loss = np.inf
    patience = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_params = model.state_dict()
    for epoch in range(args.epochs):
        train_loss = 0.0
        torch.set_grad_enabled(True)
        optimizer.zero_grad()
        z, neg_z, summary = model(expr, edge_list)
        loss = model.loss(z, neg_z, summary)

        if args.spatial:
            coords = torch.tensor(adata_filtered.obsm['spatial']).float().to(args.device)
            if subsetting:
                    cell_random_subset_1, cell_random_subset_2 = torch.randint(0, z.shape[0], (random_subset_size,)).to(
                        args.device), torch.randint(0, z.shape[0], (random_subset_size,)).to(args.device)
                    z1, z2 = torch.index_select(z, 0, cell_random_subset_1), torch.index_select(z, 0, cell_random_subset_2)
                    c1, c2 = torch.index_select(coords, 0, cell_random_subset_1), torch.index_select(coords, 0,
                                                                                                     cell_random_subset_1)
                    pdist = torch.nn.PairwiseDistance(p=2)

                    z_dists = pdist(z1, z2)
                    z_dists = z_dists / torch.max(z_dists)

                    sp_dists = pdist(c1, c2)
                    sp_dists = sp_dists / torch.max(sp_dists)
                    n_items = z_dists.size(dim=0)
            else:
                z_dists = torch.cdist(z, z, p=2)
                z_dists = torch.div(z_dists, torch.max(z_dists)).to(args.device)
                sp_dists = torch.cdist(coords, coords, p=2)
                sp_dists = torch.div(sp_dists, torch.max(sp_dists)).to(args.device)
                n_items = z.size(dim=0) * z.size(dim=0)
            penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(args.device)

            all_genes = np.char.capitalize(np.array(adata_filtered.var_names).astype(str))
            ligand = "Fgf3"
            receptor = "Fgfr3"
            ligand_index = np.where(all_genes==ligand)[0][0]
            receptor_index = np.where(all_genes == receptor)[0][0]
            # coexpression = expr[:, ligand_index] * expr[:, receptor_index]
            left_cells_expr, right_cells_expr = torch.index_select(expr, 0, cell_random_subset_1), torch.index_select(expr, 0, cell_random_subset_2)
            coexpression = left_cells_expr[:, ligand_index] * right_cells_expr[:, receptor_index]
            penalty_2 = torch.div(torch.sum(torch.mul(z_dists, coexpression)), n_items).to(args.device)
            #penalty_2 = torch.div(torch.sum(torch.mul(z_dists, 1.0 - sp_dists)), z_dists.size(dim=0)).to(args.device)
            #penalty_3 = torch.div(torch.sum(torch.mul(z_dists, sp_dists)), z_dists.size(dim=0)).to(args.device)
            loss = loss + args.penalty_scaler * penalty_1 + 2.0 * penalty_2

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if train_loss > min_loss:
            patience += 1
        else:
            patience = 0
            min_loss = train_loss
            best_params = model.state_dict()
        if epoch % 10 == 1:
            print("Epoch %d/%d" % (epoch + 1, args.epochs))
            print("Loss:" + str(train_loss))
        if patience > args.patience and epoch > args.min_stop:
            break
    model = get_model(args, expr.shape[1]).to(args.device)
    model.load_state_dict(best_params)

    z, _, _ = model(expr, edge_list)
    return z.cpu().detach().numpy()