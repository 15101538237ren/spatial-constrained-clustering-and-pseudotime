import torch
import argparse
parser = argparse.ArgumentParser()

# params to change
parser.add_argument('--gpu', type=int, default=7)
parser.add_argument('--arch', type=str, default='DGI')
parser.add_argument('--spatial', type=bool, default=True)
parser.add_argument('--n_neighbors_for_knn_graph', type=int, default=10)
parser.add_argument('--alpha_n_layer', type=int, default=1)
parser.add_argument('--z_dim', type=int, default=50)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log', type=bool, default=True)
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--min_stop', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--penalty_scaler', type=float, default=.1)
parser.add_argument('--dataset_dir', type=str, default='../data')
parser.add_argument('--output_dir', type=str, default='../output')

args = parser.parse_args()
args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'
print(args)
