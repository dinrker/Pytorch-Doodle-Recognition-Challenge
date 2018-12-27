import argparse
import os
import sys
import warnings
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

from simplified import Dataset_csv_test
from shuffle_csv_val import Simplified

# Example:
# python submit.py --arch res34 --resume res34_checkpoint_3.pth.tar

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--root', default='/home/jun/quick-draw/input/csv/train_simplified', type=str, metavar='R',
                    help='path to simplified csvs')
parser.add_argument('--testdir', default='/home/jun/quick-draw/input/test_simplified.csv', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--pretrained_dir', dest='/home/jun/quick-draw/pretrained-models', type=str, metavar='DIR',
                    help='path to pretrained models')
parser.add_argument('--arch', '-a', default='res18',type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--input_size', default=64, type=int, metavar='I',
                    help='size of input image')
parser.add_argument('--mult', default=1.0, type=float, metavar='M',
                    help='network width multiplier')
parser.add_argument('--num_classes', default=340, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--prob', default=0, type=int, metavar='P',
                    help='calculate probability')
parser.add_argument('--topk', default=3, type=int, metavar='T',
                    help='number of predictions')

def main():
    
    global args
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
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
#     elif args.distributed:
#         model.cuda()
#         model = torch.nn.parallel.DistributedDataParallel(model)
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
       
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    cudnn.benchmark = True
    
    transform_train = transforms.Compose([transforms.ToTensor()])
    
    test_loader = torch.utils.data.DataLoader(
        Dataset_csv_test(args.testdir,size=args.input_size,transform=transform_train),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    s = Simplified(args.root)
    class_name = s.list_all_categories()
    
    class_name2 = []
    for name in class_name:
        class_name2.append(name.replace(' ', '_'))

    sub_name = './sub/sub_%s-%sx%s_epoch%s' % (args.arch, args.input_size, args.input_size, str(checkpoint['epoch']))
    print(sub_name)
    
    if args.prob:
        df = test_prob(test_loader, model, class_name2, topk=args.topk)
        df.to_csv(sub_name + '_prob_top%s.csv' % args.topk, index=False)
        # df.to_csv('submit_prob_'+args.arch+'_'+str(checkpoint['epoch'])+'.csv', index=False)
        # df.to_csv('submit_'+args.arch+'_prob.csv', index=False)
    else:
        df = test(test_loader, model, class_name2, topk=args.topk)
        df.to_csv(sub_name + '_top%s.csv' % args.topk, index=False)
        # df.to_csv('submit_'+args.arch+'_'+str(checkpoint['epoch'])+'.csv', index=False)
    
    
def test(test_loader, model, class_name, topk = 3):
    
    column_list = ['key_id','word']

    df = pd.DataFrame(columns = column_list)
    L = len(test_loader)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, key_id) in enumerate(test_loader):
            input = input.to('cuda')
            output = model(input)
            pred = pred_label(output, maxk=topk)
            df1 = write_submit_df(pred, key_id, class_name, topk=topk)
            df = df.append(df1)
            sys.stdout.write('\r')
            sys.stdout.write("[%-50s] %d%%" % ('='*((50*(i+1))//L), 100*(i+1)/L))
            sys.stdout.flush()
    print(' ')        
    return df


def write_submit_df(pred, key_id, class_name, topk = 3):
    column_list = ['key_id','word']
    df = pd.DataFrame(columns = column_list)
    k = [0]*pred.size(0)
    w = [' ']*pred.size(0)
    for i in range(pred.size(0)):
        word = ' '.join([class_name[pred[i,j]] for j in range(topk)])
#         df1 = pd.DataFrame([[key_id[i].data.numpy().item(), word]], columns=column_list)        
#         df = df.append(df1)
        k[i] = key_id[i].data.numpy().item()
        w[i] = word
    df['key_id'] = pd.Series(k).values
    df['word'] = pd.Series(w).values
    return df


def test_prob(test_loader, model, class_name, topk = 3):
    
    m = nn.Softmax(dim=1)
    
    column_list = ['key_id', 'word', 'top'+str(topk)+' probs']

    df = pd.DataFrame(columns = column_list)
    L = len(test_loader)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, key_id) in enumerate(test_loader):
            input = input.to('cuda')
            output = model(input)
            pred = pred_label(output, maxk=topk)
            prob = m(output)
            df1 = write_prob_df(pred, prob, class_name, key_id=key_id, topk=topk)
            df = df.append(df1)
            sys.stdout.write('\r')
            sys.stdout.write("[%-50s] %d%%" % ('='*((50*(i+1))//L), 100*(i+1)/L))
            sys.stdout.flush()
    print(' ')        
    return df


def write_prob_df(pred, prob, class_name, target = None, key_id = None, cat_ind = None, topk = 3):
    col_prob = 'top'+str(topk)+' probs'
    column_list = ['word', col_prob]
    if target is not None:
        column_list = ['truth', 'prob']+column_list
        t = [' ']*pred.size(0)
        p = [0.0]*pred.size(0)
    elif cat_ind is not None:
        column_list = ['prob']+column_list
        p = [0.0]*pred.size(0)
    if key_id is not None:
        column_list = ['key_id']+column_list
        k = [' ']*pred.size(0)
    df = pd.DataFrame(columns = column_list)
    w = [' ']*pred.size(0)
    t3 = [[0.0]*topk]*pred.size(0)
    for i in range(pred.size(0)):
        word = ' '.join([class_name[pred[i,j]] for j in range(topk)])
        topk_probs = [0.0]*topk
        for j in range(topk):
            topk_probs[j] = round(prob[i,pred[i,j]].cpu().data.numpy().item(),6)
        if target is not None:
            cat_ind = int(target[i].data.numpy().item())
            t[i] = class_name[cat_ind]
            p[i] = prob[i,cat_ind].cpu().data.numpy().item()
        elif cat_ind is not None:
            p[i] = prob[i,cat_ind].cpu().data.numpy().item()
        if key_id is not None:
            k[i] = key_id[i].data.numpy().item()
        w[i] = word    
        t3[i] = topk_probs
    if target is not None:
        df['truth'] = pd.Series(t).values
        df['prob'] = pd.Series(p).values
    elif cat_ind is not None:
        df['prob'] = pd.Series(p).values
    if key_id is not None:
        df['key_id'] = pd.Series(k).values
    df['word'] = pd.Series(w).values
    df[col_prob] = pd.Series(t3).values
    return df


def pred_label(output, maxk = 3):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = output.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        
        return pred

    
if __name__ == '__main__':
    main()