import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from mcd.networks.base_net_mcd import BaseNetMCD


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 5, 1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class Net_FeatureExtractore(nn.Module):
    def __init__(self):
        super().__init__()
        layer = [3, 64, 64, 128]
        self.conv1 = ConvBlock(layer[0], layer[1])
        self.conv2 = ConvBlock(layer[1], layer[2])
        self.conv3 = ConvBlock(layer[2], layer[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.max_pool2d(x, stride=2, kernel_size=3, padding=1)
        x = self.conv2(x)
        x = F.max_pool2d(x, stride=2, kernel_size=3, padding=1)
        x = self.conv3(x)
        return x


class Net_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(5 * 5 * 50, 500)
        self.fc1 = nn.Linear(128, 10)
        # self.fc2 = nn.Linear(500, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        return x


class Net(BaseNetMCD):
    def __init__(self, ckpt: str = None):
        super().__init__()
        self.generator = Net_FeatureExtractore()
        self.classifier_f1 = Net_Classifier()
        self.classifier_f2 = Net_Classifier()

    def forward(self, x):
        x = self.generator(x)
        x = x.view(x.shape[0], -1)
        x_f1 = self.classifier_f1(x)
        x_f2 = self.classifier_f2(x)
        return x_f1, x_f2


from abc import ABCMeta


class Config:
    class BaseChildConfig(metaclass=ABCMeta):
        pass

    class Optim(BaseChildConfig):
        ### default
        c_name = "adam"
        c_lr = 0.0002
        c_weight_decay = 0.0005
        g_name = "adam"
        g_lr = 0.0002
        g_weight_decay = 0.0005

        # c_name = "adam"
        # c_lr = 0.0002
        # c_weight_decay = 0.0005
        # g_name = "adam"
        # g_lr = 0.0002
        # g_weight_decay = 0.0005

        # c_name = "sgd"
        # c_lr = 2e-6
        # c_weight_decay = 0.0005
        # g_name = "sgd"
        # g_lr = 2e-6
        # g_weight_decay = 0.0005

    def __init__(self):
        self.N_repeat_generator_update = 4
        # self.N_repeat_generator_update = 1
        self.multiply_loss_discrepancy = 1.8
        # self.multiply_loss_discrepancy = 4
        self.seed = 1
        # self.batch_size = 4056
        self.batch_size = 2048
        # self.test_batch_size = 64
        self.test_batch_size = self.batch_size
        self.epochs = 10000
        self.log_interval = 10
        self.save_model = True
        self.optim = self.Optim()


class Device:
    def __init__(self, N_GPUs: int):
        self.N_GPUs = N_GPUs
        self.is_cuda = False
        if not torch.cuda.is_available():
            dev = "cpu"
        elif self.N_GPUs <= 0:
            dev = "cpu"
        else:
            dev = "cuda"
            self.is_cuda = True

        self.__device = torch.device(dev)

    def get(self):
        return self.__device


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __getitem__(self, idx):
        source_idx = idx % len(self.source)
        target_idx = idx % len(self.target)
        # source_idx = idx
        # target_idx = idx
        return self.source[source_idx], self.target[target_idx]

    def __len__(self):
        return max(len(self.source), len(self.target))
        # return min(len(self.source), len(self.target))


class ConcatGray2RGB:
    def __call__(self, img: torch.Tensor):
        if not len(img.shape) == 3:
            raise ValueError
        img = torch.cat((img, img, img), 0)
        return img


class DebugMNIST(datasets.MNIST):
    def __init__(self, DATA_ROOT, img_size, batch_size):
        super().__init__(
            str(DATA_ROOT), train=True, download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                ConcatGray2RGB(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                )
            ])
        )
        self.batch_size = batch_size

    def __len__(self):
        return 2 * int(self.batch_size)


class TwoDomainDataLoader:
    def __init__(self, device: Device, batch_size: int, test_batch_size: int, seed=1):
        img_size = (32, 32)
        kwargs = {'num_workers': 1, 'pin_memory': True} if device.is_cuda else {}

        DATA_ROOT = "/raid/pytorch"
        source = datasets.SVHN(
            str(DATA_ROOT), split="train", download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                )
            ])
        )
        target = datasets.MNIST(
            str(DATA_ROOT), train=True, download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                ConcatGray2RGB(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                )
            ])
        )

        # target = DebugMNIST(DATA_ROOT=DATA_ROOT, img_size=img_size, batch_size=batch_size)
        def worker_init_fn(worker_id):
            random.seed(seed)

        train_dataset = ConcatDataset(source, target)
        self.train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            worker_init_fn=worker_init_fn, **kwargs)

        val_loader = lambda dataset: torch.utils.data.DataLoader(
            dataset, batch_size=test_batch_size,
            # shuffle=True, worker_init_fn=worker_init_fn, **kwargs)
            shuffle=False, worker_init_fn=worker_init_fn, **kwargs)
        self.val_src = val_loader(source)
        self.val_tgt = val_loader(target)
        # self.val = torch.utils.data.DataLoader(
        #     val_dataset, batch_size=test_batch_size,
        #     # shuffle=True, worker_init_fn=worker_init_fn, **kwargs)
        #     shuffle=False, worker_init_fn=worker_init_fn, **kwargs)
        # from copy import deepcopy
        # self.val = deepcopy(self.train)

    @property
    def train_len(self):
        return len(self.train.dataset)

    @property
    def val_len(self):
        return len(self.val.dataset)


import torch.optim.lr_scheduler


class LR_Scheduler:
    class NoneScheduler(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self):
            pass

        def step(self):
            pass

    EXP = "exp"
    STP = "step"
    COS = "cos"

    def __init__(self, optimizer, name):
        lrs = torch.optim.lr_scheduler
        gen = optimizer.generator
        cl1 = optimizer.classifier_f1
        cl2 = optimizer.classifier_f2
        if name == self.EXP:
            self.generator = lrs.ExponentialLR(gen, gamma=0.9)
            self.classifier_f1 = lrs.ExponentialLR(cl1, gamma=0.9)
            self.classifier_f2 = lrs.ExponentialLR(cl2, gamma=0.9)
        elif name == self.STP:
            self.generator = lrs.StepLR(gen, step_size=10, gamma=0.9)
            self.classifier_f1 = lrs.StepLR(cl1, step_size=10, gamma=0.9)
            self.classifier_f2 = lrs.StepLR(cl2, step_size=10, gamma=0.9)
        elif name == self.COS:
            self.generator = lrs.CosineAnnealingLR(gen, 5e-6, 1e-10)
            self.classifier_f1 = lrs.CosineAnnealingLR(cl1, 5e-6, 1e-10)
            self.classifier_f2 = lrs.CosineAnnealingLR(cl2, 5e-6, 2e-20)
        else:
            self.generator = self.NoneScheduler()
            self.classifier_f1 = self.NoneScheduler()
            self.classifier_f2 = self.NoneScheduler()

    def step(self):
        self.generator.step()
        self.classifier_f1.step()
        self.classifier_f2.step()


class OPT_SGD(optim.SGD):
    def __init__(self, model_params, lr, weight_decay):
        super().__init__(
            model_params, momentum=0.8,
            lr=lr, weight_decay=weight_decay
        )


class OPT_Adam(optim.Adam):
    def __init__(self, model_params, lr, weight_decay):
        super().__init__(
            model_params,
            lr=lr, weight_decay=weight_decay
        )


class Optimizer:
    supported = {
        "sgd": OPT_SGD,
        "adam": OPT_Adam
    }
    SCHEDULER_NONE = None
    SCHEDULER_EXP = LR_Scheduler.EXP
    SCHEDULER_STP = LR_Scheduler.STP
    SCHEDULER_COS = LR_Scheduler.COS

    def __init__(self, model_params, cfg: Config.Optim, scheduler: str = None):
        model_params = model_params

        if not cfg.c_name in self.supported.keys():
            raise ValueError
        if not cfg.g_name in self.supported.keys():
            raise ValueError

        self.generator = self.supported[cfg.g_name](
            model_params[0],
            lr=cfg.g_lr, weight_decay=cfg.g_weight_decay
        )
        self.classifier_f1 = self.supported[cfg.c_name](
            model_params[1],
            lr=cfg.c_lr, weight_decay=cfg.c_weight_decay
        )
        self.classifier_f2 = self.supported[cfg.c_name](
            model_params[2],
            lr=cfg.c_lr, weight_decay=cfg.c_weight_decay
        )

        self.scheduler = LR_Scheduler(self, name=scheduler)

    def zero_grad(self):
        self.generator.zero_grad()
        self.classifier_f1.zero_grad()
        self.classifier_f2.zero_grad()

    def step_all(self):
        self.step_generator()
        self.step_classifier()

    def zero_all(self):
        self.zero_generator()
        self.zero_classifier()

    def step_classifier(self):
        self.classifier_f1.step()
        self.classifier_f2.step()

    def zero_classifier(self):
        self.classifier_f1.zero_grad()
        self.classifier_f2.zero_grad()

    def step_generator(self):
        self.generator.step()

    def zero_generator(self):
        self.generator.zero_grad()

    def end_epoch(self):
        self.scheduler.step()

    @property
    def lr(self):
        gen = self.generator.param_groups[0]["lr"]
        cl1 = self.classifier_f1.param_groups[0]["lr"]
        cl2 = self.classifier_f2.param_groups[0]["lr"]
        ret = {
            "generator": gen,
            "classifier_f1": cl1,
            "classifier_f2": cl2,
        }
        return ret


class BaseLossFn(nn.Module):
    MEAN = "mean"
    SUM = "sum"

    def __init__(self, weight=None, reduction=MEAN):
        super().__init__()
        self.weight = weight
        self.reduction = reduction


class Diff2d(BaseLossFn):
    def forward(self, inputs1, inputs2):
        # return torch.mean(torch.abs(F.softmax(inputs1, dim=1) - F.softmax(inputs2, dim=1)))
        loss = torch.abs(F.softmax(inputs1, dim=1) - F.softmax(inputs2, dim=1))
        if self.reduction == self.MEAN:
            return torch.mean(loss)
        elif self.reduction == self.SUM:
            return loss
        else:
            raise ValueError


class Symkl2d(BaseLossFn):
    def forward(self, inputs1, inputs2):
        self.prob1 = F.softmax(inputs1, dim=1)
        self.prob2 = F.softmax(inputs2, dim=1)
        self.log_prob1 = F.log_softmax(self.prob1, dim=1)
        self.log_prob2 = F.log_softmax(self.prob2, dim=1)
        kl1 = F.kl_div(self.log_prob1, self.prob2, reduction=self.reduction)
        kl2 = F.kl_div(self.log_prob2, self.prob1, reduction=self.reduction)
        loss = (kl1 + kl2) * 0.5
        return loss


class Criterion(nn.Module):
    CEL = "cross_entropy"

    ### for discrepancy
    diff = "diff"
    kl = 'kl'
    symkl = 'symkl'

    def __init__(self, name=CEL, reduction="mean"):
        super().__init__()
        self.name = name
        if name == self.CEL:
            self.loss_func = nn.CrossEntropyLoss(reduction=reduction)
        elif name == self.diff:
            self.loss_func = Diff2d(reduction=reduction)
        elif name == self.symkl:
            reduction = 'batchmean' if reduction == 'mean' else reduction
            self.loss_func = Symkl2d(reduction=reduction)
        elif name == self.kl:
            reduction = 'batchmean' if reduction == 'mean' else reduction
            self.loss_func = nn.KLDivLoss(reduction=reduction)
        else:
            msg = f"NOT Supported: {name}"
            raise ValueError(msg)

    # def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.loss_func(output, target)
        return loss


class ClassifierMetrics:
    def update(self, key: str, idx: int, value: torch.Tensor):
        pass


class MCDUpdater:
    def __init__(self, cfg: Config, model: BaseNetMCD,
                 dataloader: TwoDomainDataLoader,
                 device: Device,
                 optimizer: Optimizer,
                 criterion_c: Criterion,
                 criterion_g: Criterion,
                 metrics: ClassifierMetrics):
        """
        c ≡ classifier
        g ≡ generator
        """
        self.model = model
        self.dataloader = dataloader
        self.iter = 0
        self.device = device
        self.optimizer = optimizer
        self.criterion_c = criterion_c
        self.criterion_g = criterion_g
        self.multiply_loss_discrepancy = cfg.multiply_loss_discrepancy
        self.metrics = metrics
        self.current_idx = 0

    def train(self, N_repeat_generator_update: int) -> None:
        self.model.train()

        print(self.optimizer.lr)
        pbar = tqdm(self.dataloader.train)
        for batch_idx, (source, target) in enumerate(pbar):
            src_imgs = source[0].to(self.device.get())
            src_lbls = source[1].to(self.device.get())
            tgt_imgs = target[0].to(self.device.get())

            loss_c = self._update_classifier(src_imgs, src_lbls).item()
            loss_d_max = self._maximize_discrepancy(src_imgs, src_lbls, tgt_imgs)
            loss_d_min = self._minimize_discrepancy(tgt_imgs, N_repeat=N_repeat_generator_update).item()

            msg = str(
                f"Loss | C: {loss_c:.6f}, "
                f"D MAX: {loss_d_max:.6f}, "
                f"D MIN: {loss_d_min:.6f}"
            )
            if batch_idx % 10 == 0:
                pbar.set_description(msg)

            self.current_idx += 1
        self.optimizer.end_epoch()
        self.model.save("latest_model.ckpt")
        # self.model.save("latest_model_momentumlr_2e-5.ckpt")
        # self.model.save("latest_model_lr_2e-9.ckpt")

    def _update_classifier(self, src_imgs, src_lbls):
        self.model.train_all()
        ### update generator and classifier by source samples
        self.optimizer.zero_all()
        out_f1, out_f2 = self.model(src_imgs)
        loss_f1 = self.criterion_c(out_f1, src_lbls)
        loss_f2 = self.criterion_c(out_f2, src_lbls)
        loss = loss_f1 + loss_f2
        loss.backward()
        self.optimizer.step_all()
        return loss

    def _maximize_discrepancy(self, src_imgs, src_lbls, tgt_imgs):
        self.model.train_classifier()
        self.optimizer.zero_classifier()
        out_src_f1, out_src_f2 = self.model(src_imgs)
        loss_src_f1 = self.criterion_c(out_src_f1, src_lbls).abs()
        loss_src_f2 = self.criterion_c(out_src_f2, src_lbls).abs()

        out_tgt_f1, out_tgt_f2 = self.model(tgt_imgs)
        loss_discrepancy = self.criterion_g(out_tgt_f1, out_tgt_f2).abs()
        loss_discrepancy *= 2
        loss_discrepancy *= self.multiply_loss_discrepancy
        loss = loss_src_f1 + loss_src_f2 - loss_discrepancy
        loss.backward()
        self.optimizer.step_classifier()
        return loss

    def _minimize_discrepancy(self, tgt_imgs, N_repeat):
        self.model.train_generator()
        # loss = torch.zeros((1), requires_grad=True).to(self.device.get())
        for _ in range(N_repeat):
            self.optimizer.zero_generator()
            out_tgt_f1, out_tgt_f2 = self.model(tgt_imgs)
            loss_discrepancy = self.criterion_g(out_tgt_f1, out_tgt_f2).mean().abs()
            # loss += loss_discrepancy
            loss = loss_discrepancy
            loss.backward()
        self.optimizer.step_generator()
        # loss /= N_repeat
        return loss

    def evaluate(self):
        self.model.eval()

        with torch.no_grad():
            loss_src, correct_src = self._eval(self.dataloader.val_src)
            loss_tgt, correct_tgt = self._eval(self.dataloader.val_tgt)

        val_src_len = len(self.dataloader.val_src.dataset)
        val_tgt_len = len(self.dataloader.val_tgt.dataset)

        loss_src /= val_src_len
        loss_tgt /= val_tgt_len
        mean_acc_src = correct_src / val_src_len
        mean_acc_tgt = correct_tgt / val_tgt_len

        msg = str(
            "Test set: \n"
            f"\tAverage loss: \n"
            f"\t\tsource: {loss_src:.4f}\n"
            f"\t\ttarget: {loss_tgt:.4f}\n"
            f"\tAccuracy: \n"
            f"\t\tsource: {int(correct_src)} / {val_src_len} ({100 * mean_acc_src:.2f}%)\n"
            f"\t\ttarget: {int(correct_tgt)} / {val_tgt_len} ({100 * mean_acc_tgt:.2f}%)\n"
        )
        print(msg)
        return mean_acc_src, mean_acc_tgt

    def _eval(self, dataloader):
        correct = 0
        loss = 0
        for imgs, lbls in tqdm(dataloader):
            src_imgs = imgs.to(self.device.get())
            src_lbls = lbls.to(self.device.get())

            out_f1, out_f2 = self.model(src_imgs)
            out = out_f1 + out_f2
            # out = out_f1

            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(src_lbls.view_as(pred)).sum().item()
            loss += self.criterion_c(out, src_lbls).item()
        return loss, correct


def train(seed):
    cfg = Config()
    torch.manual_seed(cfg.seed)

    device = Device(N_GPUs=2)

    dataloader = TwoDomainDataLoader(
        device=device,
        batch_size=cfg.batch_size,
        test_batch_size=cfg.test_batch_size,
        seed=seed
    )
    # from mcd.networks.resnet.resnet import ResNet
    # model = ResNet(layer_name=ResNet.NET50)
    model = Net()
    # model.load("latest_model.ckpt")
    model.to(device.get())
    optimizer = Optimizer(
        model_params=model.parameters(),
        cfg=cfg.optim,
        # scheduler=Optimizer.SCHEDULER_COS
        scheduler=Optimizer.SCHEDULER_NONE
    )
    updater = MCDUpdater(
        cfg=cfg, model=model, dataloader=dataloader,
        optimizer=optimizer, device=device,
        criterion_c=Criterion(name=Criterion.CEL),
        # criterion_g=Criterion(name=Criterion.CEL),
        criterion_g=Criterion(name=Criterion.diff),
        # criterion_g=Criterion(name=Criterion.symkl),
        # criterion_g=Criterion(name=Criterion.kl),
        metrics=ClassifierMetrics(),
    )

    from time import time
    start_time = time()
    for epoch in range(1, cfg.epochs + 1):
        print("")
        print(f"epoch: [{epoch} / {cfg.epochs}]")
        if epoch >= 10:
            cfg.N_repeat_generator_update = 4
        epoch_start_t = time()
        updater.train(N_repeat_generator_update=cfg.N_repeat_generator_update)
        updater.evaluate()
        epoch_end_t = time()
        print(f"Time => ThisEpoch {epoch_end_t - epoch_start_t:.02f} [sec] |"
              f" Total {epoch_end_t - start_time:.02f} [sec]")

    if (cfg.save_model):
        ckpt_path = "mnist_cnn.pt"
        model.save(ckpt_path)


import optuna


def objective(trial: optuna.trial.Trial):
    seed = 1

    cfg = Config()
    cfg.seed = seed
    cfg.epochs = 1
    torch.manual_seed(cfg.seed)

    device = Device(N_GPUs=2)
    dataloader = TwoDomainDataLoader(
        device=device,
        batch_size=cfg.batch_size,
        test_batch_size=cfg.test_batch_size,
        seed=seed
    )
    model = Net()
    model.to(device.get())
    ##################
    # cfg.optim.c_weight_decay = trial.suggest_loguniform("c_wd", 1e-10, 0.01)
    # cfg.optim.g_weight_decay = trial.suggest_loguniform("g_wd", 1e-10, 0.01)
    # cfg.multiply_loss_discrepancy = trial.suggest_uniform("mld", 0, 2)
    trial_opt = list(Optimizer.supported.keys())
    cfg.optim.c_name = trial.suggest_categorical("c_opt", trial_opt)
    cfg.optim.g_name = trial.suggest_categorical("g_opt", trial_opt)
    cat = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    cfg.optim.c_lr = trial.suggest_categorical("c_lr", cat)
    cfg.optim.g_lr = trial.suggest_categorical("g_lr", cat)
    # cfg.optim.c_lr = trial.suggest_loguniform("c_lr", 0.0001, 0.01)
    # cfg.optim.g_lr = trial.suggest_loguniform("g_lr", 0.00001, 0.001)

    optimizer = Optimizer(
        model_params=model.parameters(),
        cfg=cfg.optim,
        # scheduler=Optimizer.SCHEDULER_COS
        scheduler=Optimizer.SCHEDULER_NONE
    )
    updater = MCDUpdater(
        cfg=cfg, model=model, dataloader=dataloader,
        optimizer=optimizer, device=device,
        criterion_c=Criterion(name=Criterion.CEL),
        criterion_g=Criterion(name=Criterion.diff),
        metrics=ClassifierMetrics()
    )
    for epoch in range(1, cfg.epochs + 1):
        print(f"epoch: [{epoch} / {cfg.epochs}]")
        if epoch >= 10:
            cfg.N_repeat_generator_update = 4
        updater.train(N_repeat_generator_update=cfg.N_repeat_generator_update)
        _, acc = updater.evaluate()
    ret = 1 - acc
    return ret


if __name__ == '__main__':
    import random
    import numpy as np

    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    isOptuna = False
    # isOptuna = True

    if not isOptuna:
        train(seed)
    else:

        # study = optuna.create_study(direction="maximize")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=200, n_jobs=-1)

        ##########
        # oputna log
        ##########
        import json

        print("")
        print("++++ optuna trial ++++")
        print('Number of finished trials: ', len(study.trials))
        print('Best trial:')
        trial = study.best_trial
        print('  Value: ', trial.value)
        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))
        print('  User attrs:')
        for key, value in trial.user_attrs.items():
            print('    {}: {}'.format(key, value))
        with open("./best_param.json", "w") as fw:
            json.dump(study.best_params, fw)
