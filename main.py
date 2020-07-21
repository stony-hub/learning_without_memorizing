import argparse
from approch import Model


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--niter', type=int, default=20, help='number of epochs per task')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta', type=float, default=1., help='hyperparameter beta')
    parser.add_argument('--gamma', type=float, default=1., help='hyperparameter gamma')
    
    opt = parser.parse_args()
    
    model = Model()
    model.train(opt.niter, opt.lr, opt.beta, opt.gamma)


if __name__ == '__main__':
    main()
