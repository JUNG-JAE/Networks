import sys
import os
import datetime
from torch.optim.lr_scheduler import _LRScheduler
import re


"""
If your data set is highly differentiated, you can suffer from a sort of "early over-fitting". If your shuffled data happens to include a cluster of related, strongly-featured observations, your model's initial training can skew badly toward those features -- or worse, toward incidental features that aren't truly related to the topic at all.

Warm-up is a way to reduce the primacy effect of the early training examples. Without it, you may need to run a few extra epochs to get the convergence desired, as the model un-trains those early superstitions.

Many networks afford this as a command-line option. The learning rate is increased linearly over the warm-up period. If the target learning rate is p and the warm-up period is n, then the first batch iteration uses 1*p/n for its learning rate; the second uses 2*p/n, and so on: iteration i uses i*p/n, until we hit the nominal rate at iteration n.

This means that the first iteration gets only 1/n of the primacy effect. This does a reasonable job of balancing that influence.

Note that the ramp-up is commonly on the order of one epoch -- but is occasionally longer for particularly skewed data, or shorter for more homogeneous distributions. You may want to adjust, depending on how functionally extreme your batches can become when the shuffling algorithm is applied to the training set.
"""

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def get_network(args):
    if args.net == 'vgg16':
        from networks.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from networks.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from networks.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from networks.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from networks.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from networks.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from networks.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from networks.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from networks.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from networks.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from networks.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from networks.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from networks.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from networks.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from networks.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from networks.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from networks.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from networks.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from networks.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from networks.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from networks.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from networks.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from networks.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from networks.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from networks.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from networks.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from networks.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from networks.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from networks.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from networks.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from networks.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from networks.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from networks.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from networks.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from networks.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from networks.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from networks.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from networks.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from networks.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from networks.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from networks.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from networks.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from networks.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from networks.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]


def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch