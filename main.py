import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from simplified import Dataset_csv_new

import datetime
now = datetime.datetime.now()

all_start_time = time.time()


# Example:
# python main.py --arch res50 --input_size 160 --batch-size 170 --epochs 2 --lr 0.01 --linear_lr 1 --resume /home/jun/quick-draw/latest_code/models/res50-160x160-val-loss-0.5422.pth.tar


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_train', default='/home/jun/quick-draw/pytorch_11_24/data_fold0_20k', type=str, metavar='DIR',
                    help='path to training dataset')
parser.add_argument('--data_val', default='/home/jun/quick-draw/pytorch_11_24/data_val', type=str, metavar='DIR',
                    help='path to validation dataset')
parser.add_argument('--pretrained_dir', dest='/home/jun/quick-draw/pretrained-models', type=str, metavar='DIR',
                    help='path to pretrained models')

parser.add_argument('--arch', '-a', default='res18',type=str)
parser.add_argument('--num_classes', default=340, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--input_size', default=80, type=int, metavar='I',
                    help='size of input image')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--expo', default=0.8, type=float, metavar='E',
                    help='learning rate decay factor')
parser.add_argument('--mult', default=1.0, type=float, metavar='M',
                    help='network width multiplier')
parser.add_argument('--steps', default='1, 2', type=str, metavar='S',
                    help='learning_rate_steps')
parser.add_argument('--linear_lr', default=0, type=int, metavar='L',
                    help='linear learning rate policy')


best_acc1 = 0
best_loss = 10


def main():
    global args, best_acc1, best_loss
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
#     if args.pretrained:
#         print("=> using pre-trained model '{}'".format(args.arch))
#         model = models.__dict__[args.arch](pretrained=True)
#     else:
#         print("=> creating model '{}'".format(args.arch))
#         model = models.__dict__[args.arch]()

    pretrained = True if args.mult==1.0 else False
    if args.arch == 'res18':
        from models.resnet import resnet18
        model = resnet18(pretrained=pretrained,num_classes=args.num_classes,wid_mult=args.mult)
    elif args.arch == 'res34':
        from models.resnet import resnet34
        model = resnet34(pretrained=pretrained,num_classes=args.num_classes,wid_mult=args.mult)
    elif args.arch == 'res50':
        from models.resnet import resnet50
        model = resnet50(pretrained=pretrained,num_classes=args.num_classes,wid_mult=args.mult)

    elif args.arch == 'dense121':
        from models.densenet import densenet121
        model = densenet121()
    elif args.arch == 'dense161':
        from models.densenet import densenet161
        model = densenet161()
    elif args.arch == 'dense-eff':
        from models.densenet_efficient import DenseNet
        model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24))

    elif args.arch == 'mobile-v2+':
        from models.mobilenet_v2_plus.mobilenetv2plus import MobileNetV2Plus
        model = MobileNetV2Plus()
        model.load_pretrain(args.pretrained_dir + '/cityscapes_mobilenetv2_best_model.pkl')
    elif args.arch == 'se-res34':
        from models.se_resnet_34 import se_resnet34
        model = se_resnet34(num_classes=args.num_classes,wid_mult=args.mult)
        model.load_pretrain(args.pretrained_dir + '/res34_938_80_10-5-1.pth.tar')
    elif args.arch == 'seres50':
        from models.model_senet import Net
        model = Net(num_class=args.num_classes, arch='se_resnet50')
        model.load_pretrain(args.pretrained_dir + '/se_resnet50-ce0d4300.pth')
    elif args.arch == 'seresx50':
        from models.model_senet import Net
        model = Net(num_class=args.num_classes, arch='se_resnext50_32x4d')
        model.load_pretrain(args.pretrained_dir + '/se_resnext50_32x4d-a260b3a4.pth')

    elif args.arch == 'xcep':
        from models.xception import xception
        model = xception(pretrained=None)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    steps = list(map(int, args.steps.split(',')))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if 'best_loss' in checkpoint:
                best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    print('\nloading data at   %s ...  ' % time_to_str((time.time() - all_start_time)))

    # Data loading code
    traindir = args.data+'/train.csv'
    valdir = args.data_val+'/val.csv'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([transforms.ToTensor()])

    train_dataset = Dataset_csv_new(traindir,size=args.input_size,transform=transform_train)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        Dataset_csv_new(valdir,size=args.input_size,shuffle=False,transform=transform_train),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        acc1 = validate(val_loader, model, criterion)
        return

    if not os.path.isdir('logs'):
            os.mkdir('logs')
    name_time = now.strftime("%Y-%m-%d-%H-%M")
    # fname = './logs/'+args.arch+'_'+name_time
    fname = './logs/%s-%sx%s-%s_%s' % (args.arch, args.input_size, args.input_size, args.batch_size, name_time)

    f_train = open(fname+'_train.txt', 'a')
    f_val = open(fname+'_val.txt', 'a')
    f_train.write('epoch,iteration,loss,top1,top3,loss_avg,top1_avg,top3_avg\n')
    f_val.write('arch: %s ; input size: %s ; batch size: %s ; epochs: %s ; lr: %s ; linear decay: %s \n' 
        % (args.arch, args.input_size, args.batch_size, args.epochs, args.lr, args.linear_lr))
    f_val.write('resume: %s\n' % args.resume)
    f_val.write('training data: %s\n' % args.data)
    f_val.write('epoch,loss,top1,top3,map@3\n')
    
    # sname = args.arch+'_'+name_time
    sname = './models/%s-%sx%s-%s_%s' % (args.arch, args.input_size, args.input_size, args.batch_size, name_time)
    if not os.path.isdir(sname):
            os.mkdir(sname)

    print('start training at %s ...  ' % time_to_str((time.time() - all_start_time)))
    print('writing logs to    %s ...' % fname)
    
#     acc1 = validate(val_loader, model, criterion, f_val)

#     for epoch in range(args.start_epoch, args.epochs):
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.linear_lr == 0:
            adjust_learning_rate_step(optimizer, epoch, steps)

        print('\n--- [Epoch %s/%s] %s\n' % (epoch, args.epochs, '-' * 80))

        # train for one epoch
        train(train_loader, val_loader, model, criterion, optimizer, epoch, f_train, f_val, sname)

        # evaluate on validation set
        acc1, loss1 = validate(val_loader, model, criterion, f_val)


        is_best = (acc1 > best_acc1) or (loss1 < best_loss)
        best_metric=(acc1 > best_acc1, loss1 < best_loss)
        best_acc1 = max(acc1, best_acc1)
        best_loss = min(loss1, best_loss)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=sname+'/'+args.arch+'_checkpoint_epoch_{0}.pth.tar'.format(epoch), best_metric=best_metric)
        
        
    f_train.close()
    f_val.close()


def train(train_loader, val_loader, model, criterion, optimizer, epoch, f_train, f_val, sname):
    global best_acc1, best_loss

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # switch to train mode
        model.train()

        # measure data loading time
        data_time.update(time.time() - end)

        if args.linear_lr > 0:
            adjust_learning_rate_lr(optimizer, epoch, i, len(train_loader))

#         if args.gpu is not None:
#             input = input.cuda(args.gpu, non_blocking=True)
#         target = target.cuda(args.gpu, non_blocking=True)
        input, target = input.to('cuda'), target.to('cuda')

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc2, acc3 = accuracy(output, target, topk=(1,2,3))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top2.update(acc2[0], input.size(0))
        top3.update(acc3[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) or (i == len(train_loader) - 1):
            map3 = 0.5*top1.val+top2.val/6+top3.val/3
            map3_avg = 0.5*top1.avg+top2.avg/6+top3.avg/3
            
            rate = get_learning_rate(optimizer)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@3 {top3.val:.3f} ({top3.avg:.3f})\t'
                  'MAP@3 {map3:.3f} ({map3_avg:.3f})\t'
                  'Now {now_time}\t'
                  '| LR {rate: .5f}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time, now_time=time_to_str((time.time() - all_start_time)), 
                   loss=losses, top1=top1, top3=top3, map3=map3, map3_avg=map3_avg, rate=rate))

            for tem in [epoch, i, losses.val, top1.val.item(), map3.item(), losses.avg, top1.avg.item(), top3.avg.item(), map3_avg.item(), rate]:
                f_train.write(str(tem)+',')
            
            f_train.write('\n')
            
        if i > 0 and i % (args.print_freq*20) == 0:
            acc1, loss1 = validate(val_loader, model, criterion, f_val)
            if i % (args.print_freq*40) == 0:

                # remember best acc@1 and save checkpoint
                is_best = (acc1 > best_acc1) or (loss1 < best_loss)
                best_metric=(acc1 > best_acc1, loss1 < best_loss)
                best_acc1 = max(acc1, best_acc1)
                best_loss = min(loss1, best_loss)
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, filename=sname+'/'+args.arch+'_checkpoint_iter_{0}.pth.tar'.format(i), best_metric=best_metric)


def validate(val_loader, model, criterion, f_val=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
#             if args.gpu is not None:
#                 input = input.cuda(args.gpu, non_blocking=True)
#             target = target.cuda(args.gpu, non_blocking=True)
            input, target = input.to('cuda'), target.to('cuda')

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc2, acc3 = accuracy(output, target, topk=(1,2,3))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top2.update(acc2[0], input.size(0))
            top3.update(acc3[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i > 0 and i % args.print_freq == 0) or i == len(val_loader)-1:
                map3 = 0.5*top1.val+top2.val/6+top3.val/3
                map3_avg = 0.5*top1.avg+top2.avg/6+top3.avg/3
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@3 {top3.val:.3f} ({top3.avg:.3f})\t'
                      'MAP@3 {map3:.3f} ({map3_avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top3=top3, map3=map3, map3_avg=map3_avg))

        map3_avg = 0.5*top1.avg+top2.avg/6+top3.avg/3

        print(' * Loss {loss.avg:.4f} | Acc@1 {top1.avg:.3f} | Acc@3 {top3.avg:.3f} | MAP@3 {map3_avg:.3f} \n'
              .format(loss=losses, top1=top1, top3=top3, map3_avg=map3_avg))
        if f_val is not None:
            for tem in [losses.avg, top1.avg.item(), top3.avg.item(), map3_avg.item()]:
                f_val.write(str(tem)+',')
            f_val.write('\n')

    return top1.avg, losses.avg


def validate_list(val_loader, model, criterion, f_val=None):
    
    # val_list = [110, 155, 230, 430, 405, 835, 1125, 1605, 1685]
    val_list = [220, 310, 460, 860, 810, 1670, 2250, 3210, 3370]
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
#             if args.gpu is not None:
#                 input = input.cuda(args.gpu, non_blocking=True)
#             target = target.cuda(args.gpu, non_blocking=True)
            input, target = input.to('cuda'), target.to('cuda')

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc2, acc3 = accuracy(output, target, topk=(1,2,3))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top2.update(acc2[0], input.size(0))
            top3.update(acc3[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i in val_list or i == len(val_loader)-1:
                map3 = 0.5*top1.val+top2.val/6+top3.val/3
                map3_avg = 0.5*top1.avg+top2.avg/6+top3.avg/3
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'MAP@3 {map3:.3f} ({map3_avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, map3=map3, map3_avg=map3_avg))

        map3_avg = 0.5*top1.avg+top2.avg/6+top3.avg/3
        print(' * Acc@1 {top1.avg:.3f} MAP@3 {map3_avg:.3f}'
              .format(top1=top1, map3_avg=map3_avg))
        if f_val is not None:
            for tem in [losses.avg, top1.avg.item(), map3_avg.item()]:
                f_val.write(str(tem)+',')
            f_val.write('\n')

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_metric=(True, False)):

    tries=10
    assert tries > 0
    error = None
    result = None

    while tries:
        try:
            torch.save(state, filename)
            if is_best:
                # shutil.copyfile(filename, 'model_best.pth.tar')
                if best_metric[0]:
                    shutil.copyfile(filename, filename.split('iter')[0].split('epoch')[0] + 'best_acc1.pth.tar')
                if best_metric[1]:
                    shutil.copyfile(filename, filename.split('iter')[0].split('epoch')[0] + 'best_loss.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            print('model indeed saved at: %s\n' % filename)
            break

    # torch.save(state, filename)
    # if is_best:
    #     # shutil.copyfile(filename, 'model_best.pth.tar')
    #     shutil.copyfile(filename, filename.split('iter')[0] + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate_step(optimizer, epoch, steps, gamma=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in steps:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']*gamma
            param_group['lr'] = lr


def adjust_learning_rate_lr(optimizer, epoch, iter, batches):
    lr = args.lr * (1-1.0*epoch/args.epochs-1.0*iter/batches/args.epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.expo ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr


def time_to_str(t):
    t  = int(t)
    hr = t//3600
    min = int(t/60) % 60
    sec = t%60
    return '%2d hr %02d min %02d sec'%(hr,min, sec)


if __name__ == '__main__':
    main()
