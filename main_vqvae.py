import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import logging
import argparse

import torch.utils.data
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import dataset

SW = SummaryWriter(os.environ.get('AMLT_OUTPUT_DIR', './tmp'), flush_secs=30)

from models.util import setup_logging_from_args
from models.auto_encoder import *

models = {
    'custom': {'vqvae': VQ_CVAE,
               'vqvae2': VQ_CVAE2},
    'imagenet': {'vqvae': VQ_CVAE,
                 'vqvae2': VQ_CVAE2},
    'cifar10': {'vae': CVAE,
                'vqvae': VQ_CVAE,
                'vqvae2': VQ_CVAE2},
    'mnist': {'vae': VAE,
              'vqvae': VQ_CVAE},
    'shapes3d':{
        'vqvae':VQ_CVAE,
        'cae':CAE
    },
    'dsprites': {'vae': VAE,
              'vqvae': VQ_CVAE,
              'cae':CAE},
    'mpi_toy':{
        'vqvae':VQ_CVAE
    },
    'mpi_mid':{
        'vqvae':VQ_CVAE
    },
    'mpi_real':{
        'vqvae':VQ_CVAE
    },
    'clevr':{
        'vqvae':VQ_CVAE
    },
    'clevr6':{
        'vqvae':VQ_CVAE
    },
    'celebanpz':{
        'vqvae':VQ_CVAE
    },
    'celeba':{
        'vqvae':VQ_CVAE
    },
    'cars3d':{
        'vqvae' : VQ_CVAE
    },
    'object_room':{
        'vqvae':VQ_CVAE
    },
    'tetrominoes':{
        'vqvae':VQ_CVAE
    },
    'object_room_npz':{
        'vqvae':VQ_CVAE
    },
    'tetrominoes_npz':{
        'vqvae':VQ_CVAE
    },
    'coco':{
        'vqvae':VQ_CVAE
    },
    'kitti':{
        'vqvae':VQ_CVAE
    }
}
datasets_classes = {
    'custom': datasets.ImageFolder,
    'imagenet': datasets.ImageFolder,
    'cifar10': datasets.CIFAR10,
    'mnist': datasets.MNIST,
    'shapes3d': dataset.custum_dataset,
    'dsprites': dataset.custum_dataset,
    'mpi_toy': dataset.custum_dataset,
    'mpi_mid': dataset.custum_dataset,
    'mpi_real': dataset.custum_dataset,
    'clevr': dataset.custum_dataset,
    'clevr6': dataset.custum_dataset,
    'celebanpz': dataset.custum_dataset,
    'celeba': dataset.custum_dataset,
    'cars3d': dataset.custum_dataset,
    'object_room':dataset.tf_record_data,
    'tetrominoes':dataset.tf_record_data,
    'object_room_npz': dataset.custum_dataset,
    'tetrominoes_npz': dataset.custum_dataset,
    'coco': dataset.custum_dataset,
    'kitti': dataset.custum_dataset,
}
dataset_train_args = {
    'custom': {},
    'imagenet': {},
    'cifar10': {'train': True, 'download': True},
    'mnist': {'train': True, 'download': True},
    'shapes3d':{},
    'dsprites':{'name':'dsprites'},
    'mpi_toy':{'name':'mpi_toy'},
    'mpi_mid':{'name':'mpi_mid'},
    'mpi_real':{'name':'mpi_real'},
    'clevr':{'name':'clevr'},
    'clevr6': {'name':'clevr6'},
    'celebanpz': {'name':"celebanpz"},
    'celeba':{'name':"celeba"},
    'cars3d':{'name':'cars3d'},
    'object_room':{'name':'object_room'},
    'tetrominoes':{'name':'tetrominoes'},
    'object_room_npz':{'name':'object_room_npz'},
    'tetrominoes_npz':{'name':'tetrominoes_npz'},
    'coco': {'name':'coco'},
    'kitti': {'name':'kitti'}
}


dataset_test_args = {
    'custom': {},
    'imagenet': {},
    'cifar10': {'train': False, 'download': True},
    'mnist': {'train': False, 'download': True},
    'shapes3d':{},
    'dsprites':{'name':'dsprites'},
    'mpi_toy':{'name':'mpi_toy'},
    'mpi_mid':{'name':'mpi_mid'},
    'mpi_real':{'name':'mpi_real'},
    'clevr':{'name':'clevr'},
    'clevr6':{'name':'clevr6'},
    'celebanpz':{'name':'celebanpz'},
    'celeba':{'name':'celeba'},
    'cars3d':{'name':'cars3d'},
    'object_room':{'name':'object_room'},
    'tetrominoes':{'name':'tetrominoes'},
    'object_room_npz':{'name':'object_room_npz'},
    'tetrominoes_npz':{'name':'tetrominoes_npz'},
    'coco':{'name':'coco'},
    'kitti': {'name':'kitti'}
}

dataset_n_channels = {
    'custom': 3,
    'imagenet': 3,
    'cifar10': 3,
    'shapes3d':3,
    'dsprites': 1,
    'mnist': 1,
    'mpi_toy': 3,
    'mpi_mid': 3,
    'mpi_real':3,
    'clevr':3,
    'clevr6':3,
    'celebanpz':3,
    'celeba': 3,
    'cars3d': 3,
    'object_room':3,
    'tetrominoes':3,
    'object_room_npz':3,
    'tetrominoes_npz':3,
    'coco':3,
    'kitti':3,
}

dataset_transforms = {
    'custom': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'imagenet': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'cifar10': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'shapes3d': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'mnist': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5), (0.5))]),
    'dsprites': transforms.Compose([transforms.ToTensor(),
                                    lambda x: 255.*x,
                                    transforms.Normalize((0.5), (0.5))]),
    'mpi_toy': transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'mpi_mid': transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'mpi_real': transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'clevr': transforms.Compose([transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), 
    'clevr6': transforms.Compose([transforms.Resize(128),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), 
    'celebanpz': transforms.Compose([transforms.Resize(128),
                                transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'celeba': transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'cars3d': transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'object_room': lambda x: x.permute(0,3,1,2) /255. *2 - 1,
    'tetrominoes': lambda x: x.permute(0,3,1,2) /255. *2 - 1,
    'object_room_npz': transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'tetrominoes_npz': transforms.Compose([transforms.Resize(32),
                                transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'coco': transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'kitti': transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

default_hyperparams = {
    'custom': {'lr': 2e-4, 'k': 512, 'hidden': 128},
    'imagenet': {'lr': 2e-4, 'k': 512, 'hidden': 128},
    'cifar10': {'lr': 2e-4, 'k': 128, 'hidden': 256},
    'shapes3d': {'lr':2e-4, 'k':128, 'hidden': 256},
    'mnist': {'lr': 1e-4, 'k': 10, 'hidden': 64},
    'dsprites': {'lr': 1e-4, 'k': 20, 'hidden': 64},
    'mpi_toy': {'lr': 2e-4, 'k': 128, 'hidden': 256},
    'mpi_mid': {'lr':2e-4, 'k':128, 'hidden': 256},
    'mpi_real': {'lr':2e-4, 'k':128, 'hidden': 256},
    'clevr': {'lr':2e-4, 'k':128, 'hidden': 256},
    'clevr6': {'lr':2e-4, 'k':128, 'hidden': 256},
    'celebanpz': {'lr':2e-4, 'k':128, 'hidden': 256},
    'celeba': {'lr':2e-4, 'k':128, 'hidden': 256},
    'cars3d': {'lr':2e-4, 'k':128, 'hidden': 256},
    'object_room': {'lr':2e-4, 'k':128, 'hidden': 256},
    'tetrominoes': {'lr':2e-4, 'k':128, 'hidden': 128},
    'object_room_npz': {'lr':2e-4, 'k':128, 'hidden': 256},
    'coco': {'lr':2e-4, 'k':128, 'hidden': 256},
    'kitti': {'lr':2e-4, 'k':128, 'hidden': 256}

}


def main(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')

    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model', default='vae', choices=['vae', 'vqvae','cae'],
                              help='autoencoder variant to use: vae | vqvae')
    model_parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                              help='input batch size for training (default: 128)')
    model_parser.add_argument('--hidden', type=int, metavar='N',
                              help='number of hidden channels')
    model_parser.add_argument('-k', '--dict-size', type=int, dest='k', metavar='K',
                              help='number of atoms in dictionary')
    model_parser.add_argument('--lr', type=float, default=None,
                              help='learning rate')
    model_parser.add_argument('--vq_coef', type=float, default=None,
                              help='vq coefficient in loss')
    model_parser.add_argument('--commit_coef', type=float, default=None,
                              help='commitment coefficient in loss')
    model_parser.add_argument('--kl_coef', type=float, default=None,
                              help='kl-divergence coefficient in loss')

    training_parser = parser.add_argument_group('Training Parameters')
    training_parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10', 'imagenet',
                                                                          'custom', 'shapes3d','dsprites',
                                                                          "mpi_toy", 'mpi_mid', 'mpi_real',
                                                                          "clevr","celebanpz", "celeba",
                                                                          "cars3d", "object_room","tetrominoes",
                                                                          "object_room_npz", "clevr6", "coco","kitti"],
                                 help='dataset to use: mnist | cifar10 | imagenet | custom')
    training_parser.add_argument('--dataset_dir_name', default='',
                                 help='name of the dir containing the dataset if dataset == custom')
    training_parser.add_argument('--data-dir', default='/media/ssd/Datasets',
                                 help='directory containing the dataset')
    training_parser.add_argument('--epochs', type=int, default=20, metavar='N',
                                 help='number of epochs to train (default: 10)')
    training_parser.add_argument('--max-epoch-samples', type=int, default=50000,
                                 help='max num of samples per epoch')
    training_parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='enables CUDA training')
    training_parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
    training_parser.add_argument('--gpus', default='0',
                                 help='gpus used for training - e.g 0,1,3')

    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                help='how many batches to wait before logging training status')
    logging_parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results',
                                help='results dir')
    logging_parser.add_argument('--save-name', default='',
                                help='saved folder')
    logging_parser.add_argument('--data-format', default='json',
                                help='in which format to save the data')
    logging_parser.add_argument('--load_path', type=str, default='',
                                help='load path')
    args = parser.parse_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # args.cuda = False
    dataset_dir_name = args.dataset if args.dataset not in ['custom','imagenet'] else args.dataset_dir_name

    lr = args.lr or default_hyperparams[args.dataset]['lr']
    k = args.k or default_hyperparams[args.dataset]['k']
    if not args.k:
        print(args.k)
        args.k = default_hyperparams[args.dataset]['k']
    hidden = args.hidden or default_hyperparams[args.dataset]['hidden']
    num_channels = dataset_n_channels[args.dataset]

    save_path = setup_logging_from_args(args)
    writer = SummaryWriter(save_path)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    model = models[args.dataset][args.model](hidden, k=k, num_channels=num_channels)
    if args.cuda:
        model.cuda()

    if args.load_path != "":
        model.load_state_dict(torch.load(args.load_path))
        start = int(args.load_path.split("_")[-1].replace(".pth",""))
    else:
        start = 1

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10 if args.dataset == 'imagenet' else 30, 0.5,)


    if args.dataset not in ["object_room","tetrominoes"]:
        kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
        dataset_train_dir = os.path.join(args.data_dir, dataset_dir_name)
        dataset_test_dir = os.path.join(args.data_dir, dataset_dir_name)
        if args.dataset in ['imagenet', 'custom']:
            dataset_train_dir = os.path.join(dataset_train_dir, 'imagenet_train/')
            dataset_test_dir = os.path.join(dataset_test_dir, 'ImageNet/val/')
        train_loader = torch.utils.data.DataLoader(
            datasets_classes[args.dataset](dataset_train_dir,
                                        transform=dataset_transforms[args.dataset],
                                        **dataset_train_args[args.dataset]),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets_classes[args.dataset](dataset_test_dir,
                                        transform=dataset_transforms[args.dataset],
                                        **dataset_test_args[args.dataset]),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        dataset_train_dir = os.path.join(args.data_dir, dataset_dir_name)
        dataset_test_dir = os.path.join(args.data_dir, dataset_dir_name)
        train_loader = datasets_classes[args.dataset](dataset_train_dir, batch_size = args.batch_size, shuffle=True, transform=dataset_transforms[args.dataset], **dataset_train_args[args.dataset])
        test_loader = datasets_classes[args.dataset](dataset_test_dir, batch_size = args.batch_size, shuffle=False, transform=dataset_transforms[args.dataset], **dataset_test_args[args.dataset])

    for epoch in range(start, args.epochs + 1):
        train_losses = train(epoch, model, train_loader, optimizer, args.cuda,
                             args.log_interval, save_path, args, writer)
        test_losses = test_net(epoch, model, test_loader, args.cuda, save_path, args, writer)

        for k in train_losses.keys():
            name = k.replace('_train', '')
            train_name = k
            test_name = k.replace('train', 'test')
            writer.add_scalars(name, {'train': train_losses[train_name],
                                      'test': test_losses[test_name],
                                      })
        scheduler.step()


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path, args, writer):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    batch_idx, data = None, None
    for batch_idx, (data, _) in enumerate(train_loader):
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data)
        loss = model.loss_function(data, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])

        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            for k, v in losses.items():
                SW.add_scalar(f'loss/{k}', v, epoch)
            logging.info('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                         ' {time:3.2f}   {loss}'
                         .format(epoch=epoch, batch=batch_idx * len(data), total_batch=len(train_loader) * len(data),
                                 percent=int(100. * batch_idx / len(train_loader)),
                                 time=time.time() - start_time,
                                 loss=loss_string))
            start_time = time.time()
            # logging.info('z_e norm: {:.2f}'.format(float(torch.mean(torch.norm(outputs[1][0].contiguous().view(256,-1),2,0)))))
            # logging.info('z_q norm: {:.2f}'.format(float(torch.mean(torch.norm(outputs[2][0].contiguous().view(256,-1),2,0)))))
            for key in latest_losses:
                losses[key + '_train'] = 0
        if batch_idx == (len(train_loader) - 1):
            save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_train')

            write_images(data, outputs, writer, 'train')

        if args.dataset in ['imagenet', 'custom', 'object_room', 'tetrominoes'] and batch_idx * len(data) > len(train_loader)* len(data):
            print([batch_idx * len(data),len(train_loader)* len(data)])
            break

    for key in epoch_losses:
        if args.dataset not in ['imagenet', 'custom', 'object_room', 'tetrominoes']:
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
        else:
            epoch_losses[key] /= len(train_loader)
    loss_string = '\t'.join(['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    if outputs[3] != None:
        writer.add_histogram('dict frequency', outputs[3], bins=range(args.k + 1))
        model.print_atom_hist(outputs[3])
    return epoch_losses


def test_net(epoch, model, test_loader, cuda, save_path, args, writer):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    i, data = None, None
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if cuda:
                data = data.cuda()
            outputs = model(data)
            model.loss_function(data, *outputs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
            if i == 0:
                write_images(data, outputs, writer, 'test')

                save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_test')
                save_checkpoint(model, epoch, save_path)
            if args.dataset in ['imagenet', 'custom', 'object_room', 'tetrominoes', 'object_room_npz', 'tetrominoes_npz',"clevr6"] and i * len(data) > 200:
                break

    for key in losses:
        if args.dataset not in ['imagenet', 'custom', 'object_room', 'tetrominoes', 'object_room_npz', 'tetrominoes_npz',"clevr6"]:
            losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
        else:
            losses[key] /= (i * len(data))
    loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
    logging.info('====> Test set losses: {}'.format(loss_string))
    return losses


def write_images(data, outputs, writer, suffix):
    original = data.mul(0.5).add(0.5)
    original_grid = make_grid(original[:6])
    writer.add_image(f'original/{suffix}', original_grid)
    reconstructed = outputs[0].mul(0.5).add(0.5)
    reconstructed_grid = make_grid(reconstructed[:6])
    writer.add_image(f'reconstructed/{suffix}', reconstructed_grid)


def save_reconstructed_images(data, epoch, outputs, save_path, name):
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    comparison = torch.cat([data[:n],
                            outputs.view(batch_size, size[1], size[2], size[3])[:n]])
    save_image(comparison.cpu(),
               os.path.join(save_path, name + '_' + str(epoch) + '.png'), nrow=n, normalize=True)


def save_checkpoint(model, epoch, save_path):
    os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(save_path, 'checkpoints', f'model_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main(sys.argv[1:])
