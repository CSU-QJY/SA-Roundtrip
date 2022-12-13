import argparse
from main import traing

parser = argparse.ArgumentParser('')
parser.add_argument('--model', type=str, default='SA')
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--x', type=int, default=(1, 1, 128))
parser.add_argument('--y', type=int, default=(32, 32, 1))
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--n_d', type=int, default=2, help='D每训练几步G训练一步')
parser.add_argument('--epochs', type=int, default=2500)
parser.add_argument('--epoch_decay', type=int, default=200)
parser.add_argument('--alpha', type=float, default=10)
parser.add_argument('--beta', type=float, default=10)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--adam_beta_1', type=float, default=0.5)
parser.add_argument('--adam_beta_2', type=float, default=0.9)
args = parser.parse_args()

if __name__ == '__main__':
    # for i in range(15):
    #     args.x = (i + 1) * 4
    traing(args)
