'''
Siamese Neural Network for One-Shot Image Recognition
@Author: qiz19014
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import time
from random import Random
import Augmentor
from utils import AverageMeter, get_num_model
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
import shutil
import numpy as np
import random
from config import get_config
from utils import prepare_dirs, save_config, load_config



# load the dataset
def get_train_valid_loader(data_dir,
                           batch_size,
                           num_train,
                           augment,
                           way,
                           trials,
                           shuffle=False,
                           seed=0,
                           num_workers=4,
                           pin_memory=True):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    train_dataset = dset.ImageFolder(root=train_dir)
    train_dataset = OmniglotTrain(train_dataset, num_train, augment)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_dataset = dset.ImageFolder(root=valid_dir)
    valid_dataset = OmniglotTest(
        valid_dataset, trials=trials, way=way, seed=seed,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=way, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)

def get_test_loader(data_dir,
                    way,
                    trials,
                    seed=0,
                    num_workers=4,
                    pin_memory=False):
    test_dir = os.path.join(data_dir, 'test')
    test_dataset = dset.ImageFolder(root=test_dir)
    test_dataset = OmniglotTest(
        test_dataset, trials=trials, way=way, seed=seed,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=way, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return test_loader

class OmniglotTrain(Dataset):
    def __init__(self, dataset, num_train, augment=False):
        super(OmniglotTrain, self).__init__()
        self.dataset = dataset
        self.num_train = num_train
        self.augment = augment

    def __len__(self):
        return self.num_train

    def __getitem__(self, index):
        image1 = random.choice(self.dataset.imgs)

        # get image from same class
        label = None
        if index % 2 == 1:
            label = 1.0
            while True:
                image2 = random.choice(self.dataset.imgs)
                if image1[1] == image2[1]:
                    break
        # get image from different class
        else:
            label = 0.0
            while True:
                image2 = random.choice(self.dataset.imgs)
                if image1[1] != image2[1]:
                    break
        image1 = Image.open(image1[0])
        image2 = Image.open(image2[0])
        image1 = image1.convert('L')
        image2 = image2.convert('L')

        # apply transformation on the fly
        if self.augment:
            p = Augmentor.Pipeline()
            p.rotate(probability=0.5, max_left_rotation=15, max_right_rotation=15)
            p.random_distortion(
                probability=0.5, grid_width=6, grid_height=6, magnitude=10,
            )
            trans = transforms.Compose([
                p.torch_transform(),
                transforms.ToTensor(),
            ])
        else:
            trans = transforms.ToTensor()

        image1 = trans(image1)
        image2 = transforms.ToTensor()(image2)
        y = torch.from_numpy(np.array([label], dtype=np.float32))
        return (image1, image2, y)

class OmniglotTest(Dataset):
    def __init__(self, dataset, trials, way, seed=0):
        super(OmniglotTest, self).__init__()
        self.dataset = dataset
        self.trials = trials
        self.way = way
        self.transform = transforms.ToTensor()
        self.seed = seed

    def __len__(self):
        return (self.trials * self.way)

    def __getitem__(self, index):
        self.rng = Random(self.seed + index)

        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.img1 = self.rng.choice(self.dataset.imgs)
            while True:
                img2 = self.rng.choice(self.dataset.imgs)
                if self.img1[1] == img2[1]:
                    break
        # generate image pair from different class
        else:
            while True:
                img2 = self.rng.choice(self.dataset.imgs)
                if self.img1[1] != img2[1]:
                    break

        img1 = Image.open(self.img1[0])
        img2 = Image.open(img2[0])
        img1 = img1.convert('L')
        img2 = img2.convert('L')
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2

# Define the model
class SiameseNet(nn.Module):

    def __init__(self):
        super(SiameseNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_in')

    def sub_forward(self, x):
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = F.relu(F.max_pool2d(self.conv3(out), 2))
        out = F.relu(self.conv4(out))

        out = out.view(out.shape[0], -1)
        out = F.sigmoid(self.fc1(out))
        return out

    def forward(self, x1, x2):

        # encode image pairs
        h1 = self.sub_forward(x1)
        h2 = self.sub_forward(x2)

        # compute l1 distance
        diff = torch.abs(h1 - h2)

        # score the similarity between the 2 encodings
        scores = self.fc2(diff)
        return scores

# Define how to train the model
class Trainer(object):

    def __init__(self, config, data_loader, layer_hyperparams):

        self.config = config
        self.layer_hyperparams = layer_hyperparams

        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = self.valid_loader.dataset.trials
        else:
            self.test_loader = data_loader
            self.num_test = self.test_loader.dataset.trials

        self.model = SiameseNet()
        if config.use_gpu:
            self.model.cuda()

        # model params
        self.num_params = sum(
            [p.data.nelement() for p in self.model.parameters()]
        )
        self.num_model = get_num_model(config)
        self.num_layers = len(list(self.model.children()))

        print('[*] Number of model parameters: {:,}'.format(self.num_params))

        # path params
        self.ckpt_dir = os.path.join(config.ckpt_dir, self.num_model)
        self.logs_dir = os.path.join(config.logs_dir, self.num_model)

        # misc params
        self.resume = config.resume
        self.use_gpu = config.use_gpu
        self.dtype = (
            torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        )

        # optimization params
        self.best = config.best
        self.best_valid_acc = 0.
        self.epochs = config.epochs
        self.start_epoch = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.counter = 0

        # grab layer-wise hyperparams
        self.init_lrs = self.layer_hyperparams['layer_init_lrs']
        self.init_momentums = [config.init_momentum]*self.num_layers
        self.end_momentums = self.layer_hyperparams['layer_end_momentums']
        self.l2_regs = self.layer_hyperparams['layer_l2_regs']

        # compute temper rate for momentum
        if self.epochs == 1:
            f = lambda max, min: min
        else:
            f = lambda max, min: (max - min) / (self.epochs-1)
        self.momentum_temper_rates = [
            f(x, y) for x, y in zip(self.end_momentums, self.init_momentums)
        ]

        # set global learning rates and momentums
        self.lrs = self.init_lrs
        self.momentums = self.init_momentums

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=3e-4, weight_decay=6e-5,
        )

    def train(self):
        if self.resume:
            self.load_checkpoint(best=False)

        # switch to train mode
        self.model.train()

        # create train and validation log files
        optim_file = open(os.path.join(self.logs_dir, 'optim.csv'), 'w')
        train_file = open(os.path.join(self.logs_dir, 'train.csv'), 'w')
        valid_file = open(os.path.join(self.logs_dir, 'valid.csv'), 'w')

        print("\n[*] Train on {} sample pairs, validate on {} trials".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.start_epoch, self.epochs):
            print('\nEpoch: {}/{}'.format(epoch+1, self.epochs))

            train_loss = self.train_one_epoch(epoch, train_file)
            valid_acc = self.validate(epoch, valid_file)

            # check for improvement
            is_best = valid_acc > self.best_valid_acc
            msg = "train loss: {:.3f} - val acc: {:.3f}"
            if is_best:
                msg += " [*]"
                self.counter = 0
            print(msg.format(train_loss, valid_acc))

            # checkpoint the model
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model_state': self.model.state_dict(),
                    'optim_state': self.optimizer.state_dict(),
                    'best_valid_acc': self.best_valid_acc,
                }, is_best
            )
        # release resources
        optim_file.close()
        train_file.close()
        valid_file.close()

    def train_one_epoch(self, epoch, file):
        train_batch_time = AverageMeter()
        train_losses = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x1, x2, y) in enumerate(self.train_loader):
                if self.use_gpu:
                    x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
                x1, x2, y = Variable(x1), Variable(x2), Variable(y)

                # split input pairs along the batch dimension
                batch_size = x1.shape[0]

                out = self.model(x1, x2)
                loss = F.binary_cross_entropy_with_logits(out, y)

                # compute gradients and update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # store batch statistics
                toc = time.time()
                train_losses.update(loss.data, batch_size)
                train_batch_time.update(toc-tic)
                tic = time.time()

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f}".format(
                            train_batch_time.val,
                            train_losses.val,
                        )
                    )
                )
                pbar.update(batch_size)

                # log loss
                iter = (epoch * len(self.train_loader)) + i
                file.write('{},{}\n'.format(
                    iter, train_losses.val)
                )

            return train_losses.avg

    def validate(self, epoch, file):
        # switch to evaluate mode
        self.model.eval()

        correct = 0
        for i, (x1, x2) in enumerate(self.valid_loader):
            if self.use_gpu:
                x1, x2 = x1.cuda(), x2.cuda()
            x1, x2 = Variable(x1, volatile=True), Variable(x2, volatile=True)

            batch_size = x1.shape[0]

            # compute log probabilities
            out = self.model(x1, x2)
            log_probas = F.sigmoid(out)

            # get index of max log prob
            pred = log_probas.data.max(0)[1][0]
            if pred == 0:
                correct += 1

        # compute acc and log
        valid_acc = (100. * correct) / self.num_valid
        iter = epoch
        file.write('{},{}\n'.format(
            iter, valid_acc)
        )
        return valid_acc

    def test(self):
        # load best model
        self.load_checkpoint(best=self.best)

        # switch to evaluate mode
        self.model.eval()

        correct = 0
        for i, (x1, x2) in enumerate(self.test_loader):
            if self.use_gpu:
                x1, x2 = x1.cuda(), x2.cuda()
            x1, x2 = Variable(x1, volatile=True), Variable(x2, volatile=True)

            batch_size = x1.shape[0]

            # compute log probabilities
            out = self.model(x1, x2)
            log_probas = F.sigmoid(out)

            # get index of max log prob
            pred = log_probas.data.max(0)[1][0]
            if pred == 0:
                correct += 1

        test_acc = (100. * correct) / self.num_test
        print(
            "[*] Test Acc: {}/{} ({:.2f}%)".format(
                correct, self.num_test, test_acc
            )
        )

    def temper_momentum(self, epoch):
        """
        This function linearly increases the per-layer momentum to
        a predefined ceiling over a set amount of epochs.
        """
        if epoch == 0:
            return
        self.momentums = [
            x + y for x, y in zip(self.momentums, self.momentum_temper_rates)
        ]
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['momentum'] = self.momentums[i]

    def decay_lr(self):
        """
        This function linearly decays the per-layer lr over a set
        amount of epochs.
        """
        self.scheduler.step()
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.lrs[i] = param_group['lr']

    def save_checkpoint(self, state, is_best):
        filename = 'model_ckpt.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = 'best_model_ckpt.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, best=False):
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = 'model_ckpt.tar'
        if best:
            filename = 'best_model_ckpt.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )

# Start the training
def main(config):

    # ensure directories are setup
    prepare_dirs(config)

    # create Omniglot data loaders
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}
    if config.is_train:
        data_loader = get_train_valid_loader(
            config.data_dir, config.batch_size,
            config.num_train, config.augment,
            config.way, config.valid_trials,
            config.shuffle, config.random_seed,
            **kwargs
        )
    else:
        data_loader = get_test_loader(
            config.data_dir, config.way,
            config.test_trials, config.random_seed,
            **kwargs
        )


    layer_hyperparams = {
            'layer_init_lrs': [],
            'layer_end_momentums': [],
            'layer_l2_regs': []
        }
    for i in range(6):
            # sample
        lr = random.uniform(1e-4, 1e-1)
        mom = random.uniform(0, 1)
        reg = random.uniform(0, 0.1)
    #
            # store
        layer_hyperparams['layer_init_lrs'].append(lr)
        layer_hyperparams['layer_end_momentums'].append(mom)
        layer_hyperparams['layer_l2_regs'].append(reg)
    try:
        save_config(config, layer_hyperparams)
    except ValueError:
        print(
                "[!] Samples already exist. Either change the model number,",
                "or delete the json file and rerun.",
                sep=' '
            )
    return


    trainer = Trainer(config, data_loader, layer_hyperparams)

    if config.is_train:
            trainer.train()
    else:
            trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)

