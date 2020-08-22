import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import tensorboard
from torch.optim import Adam
from dataloader import get_data_loader
from network import ResNet18


class Model:
    def __init__(self, class_per_task=2, ntasks=5):
        self.net_old = None
        self.net_new = ResNet18().cuda()
        print('model initialized')
        self.class_per_task = class_per_task
        self.ntasks = ntasks

        self.test_loaders = []
        self.writer = None

    def train(self, niter, lr=1e-3, beta=1, gamma=1):
        self.writer = tensorboard.SummaryWriter('log')
        print('tensorboard initialized')

        ############################################################
        # training on the first task
        ############################################################

        train_loader, test_loader = get_data_loader(0)
        print('get data of task %d' % 0)

        self.test_loaders.append(test_loader)
        opt = Adam(self.net_new.parameters(), lr=lr)
        for epoch in range(niter):
            
            self.net_new.train()

            losses, loss_Cs = [], []

            ########################################################

            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                y_pred = self.net_new(x)[:, :self.class_per_task]
                loss_C = F.cross_entropy(y_pred, y).mean()
                loss = loss_C

                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(loss.item())
                loss_Cs.append(loss_C.item())

            ########################################################

            loss = np.array(losses, dtype=np.float).mean()
            loss_C = np.array(loss_Cs, dtype=np.float).mean()
            
            print('| epoch %d, task %d, train_loss %.4f' % (epoch, 0, loss))

            self.writer.add_scalar('train_loss', loss, epoch)
            self.writer.add_scalar('train_loss_C', loss_C, epoch)

            if epoch % 2 == 0:
                self.test(epoch, self.class_per_task)

        ############################################################
        # training on the other tasks
        ############################################################

        for ntask in range(1, self.ntasks):
            # copying the old network
            self.net_new.feature = None
            self.net_old = copy.deepcopy(self.net_new)
            self.net_old.requires_grad_(False)

            # get data
            train_loader, test_loader = get_data_loader(ntask)
            self.test_loaders.append(test_loader)

            # transferring
            self.transfer(ntask, niter, train_loader, lr, beta, gamma, (ntask + 1) * self.class_per_task)
    
    def grad_cam_loss(self, feature_o, out_o, feature_n, out_n):
        batch = out_n.size()[0]
        index = out_n.argmax(dim=-1).view(-1, 1)
        onehot = torch.zeros_like(out_n)
        onehot.scatter_(-1, index, 1.)
        out_o, out_n = torch.sum(onehot * out_o), torch.sum(onehot * out_n)
        
        grads_o = torch.autograd.grad(out_o, feature_o)[0]
        grads_n = torch.autograd.grad(out_n, feature_n, create_graph=True)[0]
        weight_o = grads_o.mean(dim=(2, 3)).view(batch, -1, 1, 1)
        weight_n = grads_n.mean(dim=(2, 3)).view(batch, -1, 1, 1)
        
        cam_o = F.relu((grads_o * weight_o).sum(dim=1))
        cam_n = F.relu((grads_n * weight_n).sum(dim=1))
        
        # normalization
        cam_o = F.normalize(cam_o.view(batch, -1), p=2, dim=-1)
        cam_n = F.normalize(cam_n.view(batch, -1), p=2, dim=-1)
        
        loss_AD = (cam_o - cam_n).norm(p=1, dim=1).mean()
        return loss_AD

    def transfer(self, ntask, niter, train_loader, lr, beta, gamma, lim):
        opt = Adam(self.net_new.parameters(), lr=lr)

        for epoch in range(niter):
            
            self.net_new.train()
            self.net_old.train()

            losses, loss_Cs, loss_Ds, loss_ADs = [], [], [], []

            ########################################################
            
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                y_pred_old = self.net_old(x)[:, :lim - self.class_per_task]
                y_pred_new = self.net_new(x)[:, :lim]

                loss_C = F.cross_entropy(y_pred_new, y).mean()
                
                loss_D = F.binary_cross_entropy_with_logits(y_pred_new[:, :-self.class_per_task], y_pred_old.detach().sigmoid())
                
                loss_AD = self.grad_cam_loss(self.net_old.feature, y_pred_old, self.net_new.feature, y_pred_new[:, :-self.class_per_task])

                loss = loss_C + loss_D * beta + loss_AD * gamma

                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(loss.item())
                loss_Cs.append(loss_C.item())
                loss_Ds.append(loss_D.item())
                loss_ADs.append(loss_AD.item())

                torch.cuda.empty_cache()

            ########################################################

            loss = np.array(losses, dtype=np.float).mean()
            loss_C = np.array(loss_Cs, dtype=np.float).mean()
            loss_D = np.array(loss_Ds, dtype=np.float).mean()
            loss_AD = np.array(loss_ADs, dtype=np.float).mean()
            
            print('| epoch %d, task %d, train_loss %.4f, train_loss_C %.4f, train_loss_D %.4f, train_loss_AD %.4f' % (epoch, ntask, loss, loss_C, loss_D, loss_AD))

            self.writer.add_scalar('train_loss', loss, epoch + niter * ntask)
            self.writer.add_scalar('train_loss_C', loss_C, epoch + niter * ntask)
            self.writer.add_scalar('train_loss_D', loss_D, epoch + niter * ntask)
            self.writer.add_scalar('train_loss_AD', loss_AD, epoch + niter * ntask)

            if epoch % 2 == 0:
                self.test(epoch + niter * ntask, (ntask + 1) * self.class_per_task)

    def test(self, total_epoch, lim):
        self.net_new.eval()
        with torch.no_grad():
            cor_num, total_num = 0, 0
            for ntask, test_loader in enumerate(self.test_loaders):
                correct_num, total = 0, 0
                for x, y in test_loader:
                    x, y = x.cuda(), y.numpy()
                    # ------------------------
                    # task incremental setting
                    # y_pred = self.net_new(x)[:, ntask * self.class_per_task:(ntask + 1) * self.class_per_task].cpu().numpy()
                    # y -=  ntask * self.class_per_task
                    # ------------------------
                    # class incremental setting
                    y_pred = self.net_new(x)[:, :lim].cpu().numpy()
                    # ------------------------
                    y_pred= y_pred.argmax(axis=-1)
                    correct_num += (y_pred == y).sum()
                    total += y.shape[0]
                acc = correct_num / total * 100
                cor_num += correct_num
                total_num += total
                self.writer.add_scalar('test_acc_%d' % ntask, acc, total_epoch)
                print('test_acc_%d %.4f' % (ntask, acc))
            acc = cor_num / total_num * 100
            self.writer.add_scalar('test_acc_total', acc, total_epoch)
            print('test_acc_total %.4f' % acc)
