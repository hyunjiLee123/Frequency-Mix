from __future__ import print_function

import argparse
import os
import shutil
import time

from PIL import Image
from matplotlib import pyplot as plt

# import augmentations
import pixmix_utils as utils
import numpy as np
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms


parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--data_path', type=str, default='/home/hyunji/Documents/FreqPixMix/data', required=False, help='Path to CIFAR and CIFAR-C directories')
parser.add_argument('--use_300k', action='store_true', help='use 300K random images as aug data')
parser.add_argument('--model', '-m', type=str, default='wrn', choices=['wrn', 'densenet', 'resnext'], help='Choose architecture.')
##
parser.add_argument('--mixing_set', type=str, default='/home/hyunji/Documents/FreqPixMix/fractals_and_fvis', required=False, help='Frequency Domain Mixing set directory.')       ####change


# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning-rate', '-lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--batch-size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-wd', type=float, default=0.0005, help='Weight decay (L2 penalty).')

# WRN Architecture options
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=4, type=int, help='Widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='Dropout probability')

# PixMix options
parser.add_argument('--beta', default=3, type=int, help='Severity of mixing')
parser.add_argument('--k', default=4, type=int, help='Mixing iterations')

parser.add_argument('--aug-severity',default=3,type=int,help='Severity of base augmentation operators')
parser.add_argument('--all-ops','-all',action='store_true',help='Turn on all operations (+brightness,contrast,color,sharpness).')

# AugMix options
parser.add_argument('--mixture-width',default=3,type=int,help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--mixture-depth',default=-1,type=int,help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--no-jsd','-nj',action='store_true',help='Turn off JSD consistency loss.')

# Checkpointing options
parser.add_argument('--save', '-s', type=str, default='./snapshots', help='Folder to save checkpoints.')
parser.add_argument('--resume', '-r', type=str, default='', help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument('--print-freq', type=int, default=50, help='Training loss print frequency (batches).')

# Acceleration
parser.add_argument('--num-workers', type=int, default=4, help='Number of pre-fetching threads.')       # default = 4


args = parser.parse_args()
print(args)

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

# high: Gaussian Noise, Impulse Noise, Shot noise, Pixelate, JPEG Compression
# mid: Defocus Blur, Glass Blur, Motion Blur, Zoom Blur, Elastic Transform
# low: Brightness, Fog, Frost, Snow, Contrast

LOWCORRUPTIONS = ['brightness', 'fog', 'frost', 'snow', 'contrast']
MIDCORRUPTIONS = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'elastic_transform']
HIGHCORRUPTIONS = ['gaussian_noise', 'impulse_noise', 'shot_noise', 'pixelate', 'jpeg_compression']

CBAR_CORRUPTIONS = [
    "blue_noise_sample", "brownish_noise", "checkerboard_cutout", 
    "inverse_sparkles", "pinch_and_twirl", "ripple", "circular_motion_blur", 
    "lines", "sparkles", "transverse_chromatic_abberation"]

NUM_CLASSES = 100 if args.dataset == 'cifar100' else 10

def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))

def baseline_image_tensorize(orig, preprocess):
    tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    mixed = tensorize(orig)
    mixed = torch.clip(mixed, 0, 1)
    return normalize(mixed)


def pixmix(orig, mixing_pic, preprocess):
    mixings = utils.pixmixmixings
    # tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    tensorize = preprocess['tensorize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(orig))
    else:
        mixed = tensorize(orig)

    for _ in range(np.random.randint(args.k + 1)):
        if np.random.random() < 0.5:
            aug_image_copy = tensorize(augment_input(orig))
        else:
            aug_image_copy = tensorize(mixing_pic)

        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, args.beta)
        # mixed = torch.clip(mixed, 0, 1)
        mixed = torch.clamp(mixed, 0, 1)

    return mixed


def freqpixmix1(orig, mixing_pic, preprocess):
    pixmixmixings = utils.pixmixmixings
    tensorize = preprocess['tensorize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(orig))
    #     mixed = FreqMix(orig, mixing_pic, tensorize)          # 1
    #     mixed = FreqMix(orig, orig, tensorize)

    else:
        mixed = FreqMix(orig, mixing_pic, tensorize)            # 1-1
        # mixed = tensorize(orig)

    for _ in range(np.random.randint(args.k + 1)):

        if np.random.random() < 0.5:
            aug_image_copy = tensorize(augment_input(orig))
        else:
            aug_image_copy = tensorize(mixing_pic)

        mixed_op = np.random.choice(pixmixmixings)
        mixed = mixed_op(mixed, aug_image_copy, args.beta)
        mixed = torch.clamp(mixed, 0, 1)

    # return normalize(mixed)
    return mixed

def freqpixmix23(orig, mixing_pic, preprocess):
    pixmixmixings = utils.pixmixmixings
    tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(orig))
    else:
        mixed = tensorize(orig)

    for _ in range(np.random.randint(args.k + 1)):

        if np.random.random() < 0.5:
            # aug_image_copy = tensorize(augment_input(orig))
            aug_image_copy = FreqMix(orig, mixing_pic, tensorize)       ### 2
        else:
            aug_image_copy = tensorize(mixing_pic)
            # aug_image_copy = FreqMix(orig, mixing_pic, tensorize)     ### 3

        mixed_op = np.random.choice(pixmixmixings)
        mixed = mixed_op(mixed, aug_image_copy, args.beta)
        mixed = torch.clamp(mixed, 0, 1)

    # return normalize(mixed)
    return mixed

def FreqPixMix7(orig, mixing_pic, preprocess):
    mixings = utils.pixmixmixings
    # tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    tensorize = preprocess['tensorize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(orig))
    else:
        mixed = tensorize(orig)

    # for _ in range(np.random.randint(args.k + 1)):
        # if np.random.random() < 0.5:
        #     aug_image_copy = tensorize(augment_input(orig))
        # else:
        #     aug_image_copy = tensorize(mixing_pic)

    for _ in range(np.random.randint(args.k + 1)):
        r = np.random.random()
        if r < 1 / 3:
            aug_image_copy = tensorize(augment_input(orig))
        elif r < 2 / 3:
            aug_image_copy = tensorize(mixing_pic)
        else:
            aug_image_copy = FreqMix(orig, mixing_pic, tensorize)

        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, args.beta)
        # mixed = torch.clip(mixed, 0, 1)
        mixed = torch.clamp(mixed, 0, 1)

    return mixed

def FreqMix(orig, mixing_pic, tensorize):
    freqmixings = utils.freqmixings
    # tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    mixed = tensorize(orig)
    aug_image_copy = tensorize(mixing_pic)

    mixed_op = np.random.choice(freqmixings)
    mixed = mixed_op(mixed, aug_image_copy, args.beta)
    mixed = torch.clamp(mixed, 0, 1)

    # return normalize(mixed)         ##### FreqMix만 단독시 정규화해야함!
    return mixed


def augment_input(image):
  aug_list = utils.augmentations_all if args.all_ops else utils.augmentations
  op = np.random.choice(aug_list)
  return op(image.copy(), args.aug_severity)

class RandomImages300K(torch.utils.data.Dataset):
    def __init__(self, file, transform):
        self.dataset = np.load(file)
        self.transform = transform

    def __getitem__(self, index):
        img = self.dataset[index]
        return self.transform(img), 0

    def __len__(self):
        return len(self.dataset)

class PixMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform PixMix."""

  def __init__(self, dataset, mixing_set, preprocess):
      self.dataset = dataset
      self.mixing_set = mixing_set
      self.preprocess = preprocess

  def __getitem__(self, i):
    x, y = self.dataset[i]
    rnd_idx = np.random.choice(len(self.mixing_set))
    mixing_pic, _ = self.mixing_set[rnd_idx]
    pixmixed = pixmix(x, mixing_pic, self.preprocess)

    final_tensor = self.preprocess['normalize'](pixmixed)
    return final_tensor, y

  def __len__(self):
    return len(self.dataset)

class FreqPixMix123Dataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform PixMix."""

  def __init__(self, dataset, mixing_set, preprocess):
      self.dataset = dataset
      self.mixing_set = mixing_set
      self.preprocess = preprocess

  def __getitem__(self, i):
    x, y = self.dataset[i]
    rnd_idx = np.random.choice(len(self.mixing_set))
    mixing_pic, _ = self.mixing_set[rnd_idx]
    # pixmixed = freqpixmix23(x, mixing_pic, self.preprocess)                 ########
    pixmixed = FreqPixMix7(x, mixing_pic, self.preprocess)

    final_tensor = self.preprocess['normalize'](pixmixed)
    return final_tensor, y

  def __len__(self):
    return len(self.dataset)

class FreqMixDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, mixing_set, preprocess):
  # def __init__(self, dataset, preprocess):
      self.dataset = dataset
      self.mixing_set = mixing_set
      self.preprocess = preprocess

  def __getitem__(self, i):
    x, y = self.dataset[i]
    rnd_idx = np.random.choice(len(self.mixing_set))
    mixing_pic, _ = self.mixing_set[rnd_idx]
    # return baseline_image_tensorize(x, self.preprocess), y
    return FreqMix(x, mixing_pic, self.preprocess), y

  def __len__(self):
    return len(self.dataset)

def train(net, train_loader, optimizer, scheduler):
  """Train for one epoch."""
  net.train()
  loss_ema = 0.
  for i, (images, targets) in enumerate(train_loader):

    optimizer.zero_grad()

    images = images.cuda()
    targets = targets.cuda()
    logits = net(images)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    # if i % args.print_freq == 0:
    #   print('Train Loss {:.3f}'.format(loss_ema))


  return loss_ema


def test(net, test_loader, adv=None):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      # adversarial
      if adv:
        images = adv(net, images, targets)
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader), total_correct / len(test_loader.dataset)


def test_c(net, test_data, base_path):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  corruption_accs_l = []
  corruption_accs_m = []
  corruption_accs_h = []

  corrs = CBAR_CORRUPTIONS if 'Bar' in base_path else CORRUPTIONS
  corrs_low = CBAR_CORRUPTIONS if 'Bar' in base_path else LOWCORRUPTIONS
  corrs_mid = CBAR_CORRUPTIONS if 'Bar' in base_path else MIDCORRUPTIONS
  corrs_high = CBAR_CORRUPTIONS if 'Bar' in base_path else HIGHCORRUPTIONS

  for corruption in corrs:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
    test_loader = torch.utils.data.DataLoader(
          test_data,
          batch_size=args.eval_batch_size,
          shuffle=False,
          num_workers=args.num_workers,
          pin_memory=True)
    test_loss, test_acc = test(net, test_loader)
    corruption_accs.append(test_acc)
    # print('{}\tTest Loss {:.3f} | Test Error {:.3f}'.format(corruption, test_loss, 100 - 100. * test_acc))

  for corruption_l in corrs_low:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption_l + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader_l = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)
    test_loss_l, test_acc_l = test(net, test_loader_l)
    corruption_accs_l.append(test_acc_l)
    # print('{}\tTest Loss {:.3f} | Test Error {:.3f}'.format(corruption_l, test_loss_l, 100 - 100. * test_acc_l))
    print('{}\tTest Error {:.3f}'.format(corruption_l, 100 - 100. * test_acc_l))

  for corruption_m in corrs_mid:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption_m + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader_m = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)
    test_loss_m, test_acc_m = test(net, test_loader_m)
    corruption_accs_m.append(test_acc_m)
    # print('{}\tTest Loss {:.3f} | Test Error {:.3f}'.format(corruption_m, test_loss_m, 100 - 100. * test_acc_m))
    print('{}\tTest Error {:.3f}'.format(corruption_m, 100 - 100. * test_acc_m))


  for corruption_h in corrs_high:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption_h + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader_h = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loss_h, test_acc_h = test(net, test_loader_h)
    corruption_accs_h.append(test_acc_h)
    # print('{}\tTest Loss {:.3f} | Test Error {:.3f}'.format(corruption_h, test_loss_h, 100 - 100. * test_acc_h))
    print('{}\tTest Error {:.3f}'.format(corruption_h, 100 - 100. * test_acc_h))


  # return np.mean(corruption_accs)
  return np.mean(corruption_accs), np.mean(corruption_accs_l), np.mean(corruption_accs_m), np.mean(corruption_accs_h)


def normalize_l2(x):
  """
  Expects x.shape == [N, C, H, W]
  """
  norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
  norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
  return x / norm

class PGD(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, grad_sign=True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        # unnormalize
        bx = (bx+1)/2

        adv_bx = bx.detach()
        adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx * 2 - 1)
                loss = F.cross_entropy(logits, by, reduction='sum')
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)

        return adv_bx*2-1

def main():
  torch.manual_seed(1)
  np.random.seed(1)

  # Load datasets
  train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip()])
  mixing_set_transform = transforms.Compose(
      [transforms.Resize(36),
       transforms.RandomCrop(32)])

  to_tensor = transforms.ToTensor()
  normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
  test_transform = transforms.Compose(
      [transforms.ToTensor(), normalize])


  if args.dataset == 'cifar10':
    train_data = datasets.CIFAR10(
        os.path.join(args.data_path, 'cifar'), train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10(
        os.path.join(args.data_path, 'cifar'), train=False, transform=test_transform, download=True)
    base_c_path = os.path.join(args.data_path, 'cifar/CIFAR-10-C/')
    base_c_bar_path = os.path.join(args.data_path, 'cifar/CIFAR-10-C-Bar/')
    num_classes = 10
  else:
    train_data = datasets.CIFAR100(
        os.path.join(args.data_path, 'cifar'), train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR100(
        os.path.join(args.data_path, 'cifar'), train=False, transform=test_transform, download=True)
    base_c_path = os.path.join(args.data_path, 'cifar/CIFAR-100-C/')
    base_c_bar_path = os.path.join(args.data_path, 'cifar/CIFAR-100-C-Bar/')
    num_classes = 100

  if args.use_300k:
    mixing_set = RandomImages300K(file='300K_random_images.npy', transform=transforms.Compose(
      [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip()]))
  else:
    mixing_set = datasets.ImageFolder(args.mixing_set, transform=mixing_set_transform)
    ####
    # freq_mixing_set = datasets.ImageFolder(args.freqmix_mixing_set, transform=mixing_set_transform)
    # pixmix_mixing_set = datasets.ImageFolder(args.pixmix_mixing_set, transform=mixing_set_transform)
    # pixmix_png = datasets.ImageFolder(args.pixmix_png, transform=train_transform)

  print('train_size', len(train_data))
  # print('aug_size', len(mixing_set))

  # train_data = PixMixDataset(train_data, mixing_set, {'normalize': normalize, 'tensorize': to_tensor})
  # train_data = PNG_PixMixDataset(train_data, pixmix_png, mixing_set, {'normalize': normalize, 'tensorize': to_tensor})
  # train_data = FreqTuneDataset(train_data, {'normalize': normalize, 'tensorize': to_tensor})
  # train_data = FreqMixDataset(train_data, mixing_set, {'normalize': normalize, 'tensorize': to_tensor})
  # train_data = FreqMixPixMixDataset(train_data, freq_mixing_set, pixmix_mixing_set, {'normalize': normalize, 'tensorize': to_tensor})
  train_data = FreqPixMix123Dataset(train_data, mixing_set, {'normalize': normalize, 'tensorize': to_tensor})
  # Fix dataloader worker issue
  # https://github.com/pytorch/pytorch/issues/5059
  def wif(id):
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))

  train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True,
      worker_init_fn=wif)

  test_loader = torch.utils.data.DataLoader(
      test_data,
      batch_size=args.eval_batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)

  # Create model
  if args.model == 'densenet':
    net = densenet(num_classes=num_classes)
  elif args.model == 'wrn':
    net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
  elif args.model == 'resnext':
    net = resnext29(num_classes=num_classes)

  optimizer = torch.optim.SGD(
      net.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.decay,
      nesterov=True)

  # Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  # initialize adversary
  adversary = PGD(epsilon=2./255, num_steps=20, step_size=0.5/255).cuda()

  start_epoch = 0

  if args.resume:
    if os.path.isfile(args.resume):
      checkpoint = torch.load(args.resume)
      start_epoch = checkpoint['epoch'] + 1
      best_acc = checkpoint['best_acc']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print('Model restored from epoch:', start_epoch)

  if args.evaluate:
    # Evaluate clean accuracy first because test_c mutates underlying data
    test_loss, test_acc = test(net, test_loader)
    print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
        test_loss, 100 - 100. * test_acc))

    # adv_test_loss, adv_test_acc = test(net, test_loader, adv=adversary)
    # print('Adversarial\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
    #     adv_test_loss, 100 - 100. * adv_test_acc))

    # test_c_acc = test_c(net, test_data, base_c_path)
    test_c_acc, test_c_acc_l, test_c_acc_m, test_c_acc_h = test_c(net, test_data, base_c_path)
    print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))
    print('Mean Low Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc_l))
    print('Mean Mid Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc_m))
    print('Mean High Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc_h))
    return

  scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer,
      lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
          step,
          args.epochs * len(train_loader),
          1,  # lr_lambda computes multiplicative factor
          1e-6 / args.learning_rate))

  if not os.path.exists(args.save):
    os.makedirs(args.save)
  elif args.save != './snapshots':
    raise Exception('%s exists' % args.save)
  if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

  log_path = os.path.join(args.save,
                          args.dataset + '_' + args.model + '_training_log.csv')
  with open(log_path, 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

  best_acc = 0
  print('Beginning training from epoch:', start_epoch + 1)
  for epoch in range(start_epoch, args.epochs):
    begin_time = time.time()

    train_loss_ema = train(net, train_loader, optimizer, scheduler)
    test_loss, test_acc = test(net, test_loader)

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    checkpoint = {
        'epoch': epoch,
        'dataset': args.dataset,
        'model': args.model,
        'state_dict': net.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }

    save_path = os.path.join(args.save, 'checkpoint.pth.tar')
    torch.save(checkpoint, save_path)
    if is_best:
      shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))

    with open(log_path, 'a') as f:
      f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
          (epoch + 1),
          time.time() - begin_time,
          train_loss_ema,
          test_loss,
          100 - 100. * test_acc,
      ))

    print(
        'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
        ' Test Error {4:.2f}'
        .format((epoch + 1), int(time.time() - begin_time), train_loss_ema,
                test_loss, 100 - 100. * test_acc))

  # _, adv_test_acc = test(net, test_loader, adv=adversary)
  # print('Adversarial Test Error: {:.3f}\n'.format(100 - 100. * adv_test_acc))
  
  # test_c_acc = test_c(net, test_data, base_c_path)
  test_c_acc, test_c_acc_l, test_c_acc_m, test_c_acc_h = test_c(net, test_data, base_c_path)
  # print('Mean C Corruption Error: {:.3f}\n'.format(100 - 100. * test_c_acc))

  # test_c_bar_acc = test_c(net, test_data, base_c_bar_path)
  # print('Mean C-Bar Corruption Error: {:.3f}\n'.format(100 - 100. * test_c_bar_acc))

  # print('Mean Corruption Error: {:.3f}\n'.format(100 - 100. * (15*test_c_acc + 10*test_c_bar_acc)/25))

  # with open(log_path, 'a') as f:
  #   f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
  #           (args.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))



if __name__ == '__main__':
  main()
