import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from models.ada_conv import ConvNet, WAE, WAE_Cifar, Adversary, ResNet18, WRN_16_4
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import numpy
import random
from metann import Learner
from utils.digits_process_dataset import *
from utils import utils

parser = argparse.ArgumentParser(description='Training on Digits')
parser.add_argument('--dataset', default='mnist', type=str,
                    help='dataset mnist or cifar10')

parser.add_argument('--num_iters', default=10001, type=int, # MNIST
# parser.add_argument('--num_iters', default=1671, type=int, # PACS
                    help='number of total iterations to run')

parser.add_argument('--start_iters', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--min-learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_max', '--adv-learning-rate', default=1, type=float,
                    help='adversarial learning rate')
parser.add_argument('--gamma', default=1, type=float,
                    help='coefficient of constraint')
parser.add_argument('--beta', default=2000, type=float,
                    help='coefficient of relaxation')

parser.add_argument('--T_adv', default=25, type=int,
# parser.add_argument('--T_adv', default=5, type=int, # PACS
                    help='iterations for adversarial training')

parser.add_argument('--advstart_iter', default=0, type=int,
                    help='iterations for pre-train')

parser.add_argument('--K', default=3, type=int,
# parser.add_argument('--K', default=0, type=int,
                    help='number of augmented domains')
parser.add_argument('--T_min', default=100, type=int,
# parser.add_argument('--T_min', default=10, type=int,
                    help='intervals between domain augmentation')
parser.add_argument('--print-freq', '-p', default=1000, type=int,
# parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')

parser.add_argument('--resume', default=None, type=str,
                    help='path to saved checkpoint (default: none)')
parser.add_argument('--name', default='Digits', type=str,
                    help='name of experiment')
parser.add_argument('--mode',  default='train', type=str,
                    help='train or test')
parser.add_argument('--GPU_ID', "-g", default=0, type=int,
                    help='GPU_id')
parser.add_argument('--cov_weight', "-w", default=0.1, type=float,
                    help='neutron cov weight')
parser.add_argument('--activate_threshold', "-t", default=0.005, type=float,
                    help='neutron activation threshold')
parser.add_argument('--cover_ratio', "-r", default=1.0, type=float,
                    help='re-activation ratio')
parser.add_argument('--cov_mode', "-m", default=4, type=int,
                    help='coverage mode')
def main():
    torch.autograd.set_detect_anomaly(True)
    global args
    args = parser.parse_args()
    args.data_dir = os.path.expanduser('~/Datasets')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 on stackoverflow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_ID)

    exp_name = args.name

    kwargs = {'num_workers': 4}

    # create model, use Learner to wrap it
    model = None
    if args.dataset == 'mnist':
        model = Learner(ConvNet())
    elif args.dataset in ['photo']:
        m = ResNet18()
        m.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        model = Learner(m)
    elif args.dataset in ['cifar10']:
        model = Learner(WRN_16_4())
    else:
        print("Error")
    
    model = model.cuda()
    cudnn.benchmark = True

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['iter']
            prec = checkpoint['prec']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (iter {})"
                  .format(args.resume, checkpoint['iter']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.mode == 'train':
        train(model, exp_name, kwargs)
        evaluation(model, args.data_dir, args.batch_size, kwargs, sd=args.dataset)
    else:
        evaluation(model, args.data_dir, args.batch_size, kwargs, sd=args.dataset)

def get_lr_cifar(epoch, lr_decay_ratio):
    """
    Ref (Origin WRN implementation):
        https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/main.py#L162
    """
    if epoch in list(range(1,60)):
        return lr_decay_ratio**0
    elif epoch in list(range(60,120)):
        return lr_decay_ratio**1
    elif epoch in list(range(120,160)):
        return lr_decay_ratio**2
    elif epoch in list(range(160,201)):
        return lr_decay_ratio**3
    else:
        return 0.0

def train(model, exp_name, kwargs):
    print('Pre-train wae')
    # construct train and val dataloader
    # data shape: B X H X W X Channels
    train_loader, val_loader, img_shape = construct_datasets(args.data_dir, args.batch_size, kwargs, args.dataset)
    wae_input_dims = img_shape[1]*img_shape[2]*img_shape[3]
    wae = None
    z_dim = 20
    if args.dataset in ['cifar10']:
        wae = WAE_Cifar(wae_input_dims).cuda()
        z_dim = 512
    else:
        wae = WAE(wae_input_dims).cuda()
    wae_optimizer = torch.optim.Adam(wae.parameters(), lr=1e-3)
    discriminator = Adversary(z_dim).cuda()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    for epoch in range(1, 20 + 1):
        wae_train(wae, discriminator, train_loader, wae_optimizer, d_optimizer, epoch, z_dim)

    print('Training task model')
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    mse_loss = nn.MSELoss().cuda()

    optimizer = None
    if args.dataset in ['cifar10']:
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=.0005, momentum=.9, nesterov=True, lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # only augmented domains
    only_virtual_test_images = []
    only_virtual_test_labels = []
    train_loader_iter = iter(train_loader)
    # counter for domain augmentation
    counter_k = 0
    cur_epoch = 0
    lr_decay_ratio = 0.2

    for t in range(args.start_iters, args.num_iters):
        if t % len(train_loader) == 0:
            cur_epoch += 1
            layer_neuron_activated_dict = {}
            layer_neuron_activated_dict_mt = {}

        cur_lr = args.lr
        if cur_epoch in list(range(1,60)):
            cur_lr = args.lr * lr_decay_ratio**0
        elif cur_epoch in list(range(60,120)):
            cur_lr = args.lr * lr_decay_ratio**1
        elif cur_epoch in list(range(120,160)):
            cur_lr = args.lr * lr_decay_ratio**2
        elif cur_epoch in list(range(160,201)):
            cur_lr = args.lr * lr_decay_ratio**3
        else:
            print("Wrong Epoch!")
        for g in optimizer.param_groups:
            g['lr'] = cur_lr

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        break_point = int((len(train_loader) - 1) / (counter_k + 1))
        src_num = int(args.batch_size / (counter_k + 1))
        aug_num = args.batch_size - src_num

        # domain augmentation
        if (t > args.advstart_iter) and ((t + 1 - args.advstart_iter) % args.T_min == 0) and (counter_k < args.K):
            model.eval()
            params = list(model.parameters())
            virtual_test_images = []
            virtual_test_labels = []
            aug_start_time = time.time()

            for i, (input_a, target_a) in enumerate(train_loader):
                if i == break_point:
                    break

                if counter_k > 0:
                    input_b, target_b = next(aug_loader_iter)
                    input_comb = torch.cat((input_a[:src_num].float(), input_b[:aug_num])).cuda(non_blocking=True)
                    target_comb = torch.cat((target_a[:src_num].long(), target_b[:aug_num])).cuda(non_blocking=True)
                    input_aug = input_comb.clone()
                    target_aug = target_comb.clone()
                else:
                    input_a = input_a.cuda(non_blocking=True).float()
                    target_a = target_a.cuda(non_blocking=True).long()
                    input_aug = input_a.clone()
                    target_aug = target_a.clone()

                input_aug = input_aug.cuda(non_blocking=True)
                target_aug = target_aug.cuda(non_blocking=True)
                aug_optimizer = torch.optim.SGD([input_aug.requires_grad_()], args.lr_max)

                if counter_k == 0:
                    input_feat, output, _ = model.functional(params, False, input_a, return_feat=True)
                    recon_batch, _, = wae(input_a)
                else:
                    input_feat, output, _ = model.functional(params, False, input_comb, return_feat=True)
                    recon_batch, _, = wae(input_comb)

                # iteratively generate adversarial samples
                for n in range(args.T_adv):
                    input_aug_feat, output_aug, _ = model.functional(params, False, input_aug, return_feat=True)
                    recon_batch_aug, _, = wae(input_aug)
                    # Constraint
                    constraint = mse_loss(input_feat, input_aug_feat)
                    ce_loss = criterion(output_aug, target_aug)
                    # Relaxation
                    relaxation = mse_loss(recon_batch, recon_batch_aug)
                    adv_loss = -(args.beta * relaxation + ce_loss - args.gamma * constraint)
                    aug_optimizer.zero_grad()
                    adv_loss.backward(retain_graph=True)
                    aug_optimizer.step()

                virtual_test_images.append(input_aug.data.cpu().numpy())
                virtual_test_labels.append(target_aug.data.cpu().numpy())
            virtual_test_images, virtual_test_labels = asarray_and_reshape(virtual_test_images, virtual_test_labels, img_shape)

            if counter_k == 0:
                only_virtual_test_images = np.copy(virtual_test_images)
                only_virtual_test_labels = np.copy(virtual_test_labels)
            else:
                only_virtual_test_images = np.concatenate([only_virtual_test_images, virtual_test_images])
                only_virtual_test_labels = np.concatenate([only_virtual_test_labels, virtual_test_labels])

            # dataloader for domain augmentation
            aug_size = len(only_virtual_test_labels)
            X_aug = torch.stack([torch.from_numpy(only_virtual_test_images[i]) for i in range(aug_size)])
            y_aug = torch.stack([torch.from_numpy(np.asarray(i)) for i in only_virtual_test_labels])
            aug_dataset = torch.utils.data.TensorDataset(X_aug, y_aug)
            aug_loader = torch.utils.data.DataLoader(aug_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
            aug_loader_iter = iter(aug_loader)

            # dataloader for the latest domain augmentation
            new_aug_size = len(virtual_test_labels)
            new_X_aug = torch.stack([torch.from_numpy(virtual_test_images[i]) for i in range(new_aug_size)])
            new_y_aug = torch.stack([torch.from_numpy(np.asarray(i)) for i in virtual_test_labels])
            new_aug_dataset = torch.utils.data.TensorDataset(new_X_aug, new_y_aug)
            new_aug_loader = torch.utils.data.DataLoader(new_aug_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
            new_aug_loader_iter = iter(new_aug_loader)

            # re-train a wae on  the latest domain augmentation
            if counter_k + 1 < args.K:
                if args.dataset in ['cifar10']:
                    wae = WAE_Cifar(wae_input_dims).cuda()
                else:
                    wae = WAE(wae_input_dims).cuda()
                wae_optimizer = torch.optim.Adam(wae.parameters(), lr=1e-3)
                discriminator = Adversary(z_dim).cuda()
                d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
                for epoch in range(1, 20 + 1):
                    wae_train(wae, discriminator, new_aug_loader, wae_optimizer, d_optimizer, epoch, z_dim)
            aug_end_time = time.time()
            print('aug duration', (aug_end_time - aug_start_time) / 60)
            counter_k += 1

        model.train()
        try:
            input, target = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            input, target = next(train_loader_iter)

        input, target = input.cuda(non_blocking=True).float(), target.cuda(non_blocking=True).long()
        params = list(model.parameters())
        output, layer_output_dict = model.functional(params, True, input)

        neuron_cover_str = "None"
        if args.cov_mode == 1 or args.cov_mode == 4:
            loss = criterion(output, target)
        else:
            ### Neutron Cov Loss - BEGIN
            layer_neuron_activated_dict, total_act_nron, total_nron = utils.update_coverage_v2(
                layer_output_dict, args.activate_threshold, layer_neuron_activated_dict
            )
            neurons_2b_covered, all_not_covered  = utils.neuron_to_cover(layer_neuron_activated_dict, args.cover_ratio)
            neuron_coverage = utils.cal_neurons_cov_loss(layer_output_dict, neurons_2b_covered)
            neuron_cover_str = "cov%d-act%d" % (len(neurons_2b_covered), total_act_nron)
            loss = criterion(output, target) - args.cov_weight * neuron_coverage
            ### Neutron Cov Loss - END

        neuron_cover_str_mt = 'None'
        if counter_k == 0:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
        else:
            grads = torch.autograd.grad(loss, params, create_graph=True)
            params = [(param - args.lr * grad).requires_grad_() for param, grad in zip(params, grads)]
            try:
                input_b, target_b = next(aug_loader_iter)
            except:
                aug_loader_iter = iter(aug_loader)
                input_b, target_b = next(aug_loader_iter)

            input_b, target_b = input_b.cuda(non_blocking=True), target_b.cuda(non_blocking=True).long()
            output_b, layer_output_dict_mt = model.functional(params, True, input_b)

            neuron_cover_str_mt = 'None'
            if args.cov_mode == 2 or args.cov_mode == 4:
                loss_b = criterion(output_b, target_b)
            else:
                ### Neutron Cov Loss - BEGIN
                layer_neuron_activated_dict_mt, total_act_nron_mt, total_nron_mt = utils.update_coverage_v2(
                    layer_output_dict_mt, args.activate_threshold, layer_neuron_activated_dict_mt
                )

                neurons_2b_covered_mt, all_not_covered_mt  = utils.neuron_to_cover(layer_neuron_activated_dict_mt, args.cover_ratio)
                neuron_coverage_mt = utils.cal_neurons_cov_loss(layer_output_dict_mt, neurons_2b_covered_mt)
                neuron_cover_str_mt = "cov%d-act%d" % (len(neurons_2b_covered_mt), total_act_nron_mt)
                loss_b = criterion(output_b, target_b)  - args.cov_weight * neuron_coverage_mt
                ### Neutron Cov Loss - END

            loss_combine = (loss + loss_b) / 2
            optimizer.zero_grad()
            loss_combine.backward()

        cover_str = neuron_cover_str+'/'+neuron_cover_str_mt
        optimizer.step()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        if t % args.print_freq == 0:
            print(cover_str)
            print('Iter: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(t, t, args.num_iters, batch_time=batch_time, loss=losses, top1=top1))
            # evaluate on validation set per print_freq, compute acc on the whole val dataset
            prec1 = validate(val_loader, model)
            print("validation set acc", prec1)

            save_checkpoint({
                'iter': t + 1,
                'state_dict': model.state_dict(),
                'prec': prec1,
            }, args.dataset, exp_name)

def wae_train(model, D, new_aug_loader, optimizer, d_optimizer, epoch, z_dim):

    def sample_z(n_sample=None, dim=None, sigma=None, template=None):
        if n_sample is None:
            n_sample = 32
        if dim is None:
            dim = 20
        if sigma is None:
            sigma = z_sigma
        z = sigma * Variable(template.data.new(template.size()).normal_())
        return z

    z_var = 1
    z_sigma = math.sqrt(z_var)
    ones = Variable(torch.ones(args.batch_size, 1)).cuda()
    zeros = Variable(torch.zeros(args.batch_size, 1)).cuda()
    param = 100
    model.train()
    train_loss = 0

    input_dims = None
    for batch_idx, (data, _) in enumerate(new_aug_loader):
        input_dims = data.shape[1] * data.shape[2] * data.shape[3]
        break

    for batch_idx, (data, _) in enumerate(new_aug_loader):
        input_comb = data.cuda(non_blocking=True).float()
        optimizer.zero_grad()

        recon_batch, z_tilde = model(input_comb)
        z = sample_z(template=z_tilde, sigma=z_sigma, dim=z_dim)
        log_p_z = log_density_igaussian(z, z_var).view(-1, 1)
        D_z = D(z)
        D_z_tilde = D(z_tilde)
        D_loss = F.binary_cross_entropy_with_logits(D_z + log_p_z, ones) + \
                 F.binary_cross_entropy_with_logits(D_z_tilde + log_p_z, zeros)

        total_D_loss = param * D_loss
        d_optimizer.zero_grad()
        total_D_loss.backward(retain_graph=True)
        d_optimizer.step()

        BCE = F.binary_cross_entropy(recon_batch, input_comb.reshape((recon_batch.shape)), reduction='sum')
        Q_loss = F.binary_cross_entropy_with_logits(D_z_tilde + log_p_z, ones)
        loss = BCE + param * Q_loss
        loss.backward(retain_graph=True)
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(new_aug_loader.dataset),
                100. * batch_idx / len(new_aug_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(new_aug_loader.dataset)))

if __name__ == '__main__':
    seed_val = 9963
    print("Using seed: %d!" % seed_val)
    torch.manual_seed(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_val)
    random.seed(seed_val)
    main()
