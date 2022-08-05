import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import os
import math
from tqdm import tqdm
from sklearn.metrics import f1_score
from mlperf_logging import mllog
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from data_wrapper import SGCLDataLoader
from model_wrapper import TorchModel

# Training settings
parser = argparse.ArgumentParser(description='STEMDL Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--arch', default='resnet50',
                    help='model architecture')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.001,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')
parser.add_argument('--mixup', action='store_true', default=False,
                    help='mixup data augumentation')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='mixup interpolation coefficient')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--submission_org', default='ORNL',
                    help='mlperf submission orgnization')
parser.add_argument('--submission_division', default='open',
                    help='mlperf submission division')
parser.add_argument('--submission_status', default='onprem',
                    help='mlperf submission status')
parser.add_argument('--submission_platform', default='Summit',
                    help='mlperf submission platform')


def mixup(x, y, alpha=1.0, cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


criterion = nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    train_f1score = Metric('train_f1_score')
    iteration = 0
    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            # adjust_learning_rate(epoch, batch_idx)
            iteration += 1

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                if args.mixup:
                    data_batch, target_batch_a, target_batch_b, lam = mixup(data_batch, target_batch,
                                                                            args.alpha, args.cuda)
                    data_batch, target_batch_a, target_batch_b = map(Variable, (data_batch,
                                                                                target_batch_a, target_batch_b))

                    output = model(data_batch)
                    train_accuracy.update(lam * accuracy(output, target_batch_a)
                                          + (1 - lam) * accuracy(output, target_batch_b))
                    train_f1score.update(lam * f1score(output, target_batch_a)
                                         + (1 - lam) * f1score(output, target_batch_b))

                    loss = lam * criterion(output, target_batch_a) + (1 - lam) * criterion(output, target_batch_b)
                else:
                    output = model(data_batch)
                    train_accuracy.update(accuracy(output, target_batch))
                    train_f1score.update(f1score(output, target_batch))
                    loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item(),
                           'f1_score': 100. * train_f1score.avg.item()})
            t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        log_writer.add_scalar('train/f1_score', train_f1score.avg, epoch)


def validate(epoch):
    if hvd.rank() == 0:
        mllogger.start(key=mllog.constants.EVAL_START, metadata={"epoch_num": epoch + 1})
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')
    val_f1score = Metric('val_f1_score')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output_list = [model(torch.unsqueeze(data[:, idx, :, :], 1)) for idx in [0, 1, 2]]
                output = torch.mean(torch.stack(output_list), dim=0)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                val_f1score.update(f1score(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item(),
                               'f1_score': 100. * val_f1score.avg.item()})
                t.update(1)
    if hvd.rank() == 0:
        mllogger.event(key=mllog.constants.EVAL_ACCURACY, value=100 * val_accuracy.avg.item(), clear_line=True)
        mllogger.event(key="f1_score", value=100 * val_f1score.avg.item(), clear_line=True)
        mllogger.end(key=mllog.constants.EVAL_STOP, metadata={"epoch_num": epoch + 1})
    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)
        log_writer.add_scalar('val/f1_score', val_f1score.avg, epoch)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def f1score(output, target):
    pred = output.max(1, keepdim=True)[1]
    f1 = f1_score(target.cpu(), pred.cpu(), average="macro")
    return torch.tensor(f1)


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    mllog_file = os.path.join(args.log_dir, "stemdl_classification.log")
    mllog.config(filename=mllog_file)
    mllogger = mllog.get_mllogger()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    kwargs['multiprocessing_context'] = 'forkserver'
    kwargs['allreduce_batch_size'] = allreduce_batch_size
    train_data_wrapper = SGCLDataLoader(args.train_dir, shuffle=True, mode="train", **kwargs)
    val_data_wrapper = SGCLDataLoader(args.val_dir, shuffle=True, mode="val", **kwargs)
    train_loader = train_data_wrapper.loader
    train_sampler = train_data_wrapper.sampler
    train_dataset = train_data_wrapper.dataset
    val_loader = val_data_wrapper.loader
    val_sampler = val_data_wrapper.sampler
    val_dataset = val_data_wrapper.dataset
    # Set up standard ResNet-50 model.
    model_wrapper = TorchModel(args.arch)
    model = model_wrapper.model
    print("Model: ", model)
    # print(model.model)

    if hvd.rank() == 0:
        mllogger.event(key=mllog.constants.CACHE_CLEAR)
        mllogger.event(key=mllog.constants.SUBMISSION_BENCHMARK, value="stemdl_classification")
        mllogger.event(key=mllog.constants.SUBMISSION_ORG, value=args.submission_org)
        mllogger.event(key=mllog.constants.SUBMISSION_DIVISION, value=args.submission_division)
        mllogger.event(key=mllog.constants.SUBMISSION_STATUS, value=args.submission_status)
        mllogger.event(key=mllog.constants.SUBMISSION_PLATFORM, value=args.submission_platform)

        mllogger.start(key=mllog.constants.INIT_START)
        mllogger.event(key='number_of_ranks', value=hvd.size())
        mllogger.event(key=mllog.constants.SEED, value=args.seed)
        mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=args.batch_size * hvd.size())
        mllogger.event(key=mllog.constants.TRAIN_SAMPLES, value=len(train_dataset))
        mllogger.event(key=mllog.constants.EVAL_SAMPLES, value=len(val_dataset))
        mllogger.event(key=mllog.constants.OPT_NAME, value="AdamW")
        mllogger.event(key=mllog.constants.OPT_BASE_LR, value=args.base_lr)
        mllogger.end(key=mllog.constants.INIT_STOP)

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.AdamW(model_wrapper.params_to_update,  # model.parameters(),
                            lr=(args.base_lr *
                                lr_scaler),
                            weight_decay=args.wd)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=args.batches_per_allreduce,
        op=hvd.Adasum if args.use_adasum else hvd.Average)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    # print(model)
    if hvd.rank() == 0:
        mllogger.start(key=mllog.constants.RUN_START)
        mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=args.batch_size * hvd.size())
    for epoch in range(resume_from_epoch, args.epochs):
        if hvd.rank() == 0:
            mllogger.start(key=mllog.constants.EPOCH_START, metadata={"epoch_num": epoch + 1})
        train(epoch)
        validate(epoch)
        save_checkpoint(epoch)
        if hvd.rank() == 0:
            mllogger.start(key=mllog.constants.EPOCH_STOP, metadata={"epoch_num": epoch + 1})
    if hvd.rank() == 0:
        mllogger.end(key=mllog.constants.RUN_STOP, metadata={"status": "success"})
