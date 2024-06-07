import json
import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator
from torch.nn import functional as F
import numpy as np
import sklearn.metrics as sk
from ..data.transforms import build_transform
from ..data.data_manager import build_data_loader

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(
            'Found checkpoint in "{}". Will resume training'.format(directory)
        )

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(
                "Initializing summary writer for tensorboard "
                "with log_dir={}".format(log_dir)
            )
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)

    def prograd_backward_and_update(
        self, loss_a, loss_b, lambda_=1, names=None
    ):
        # loss_b not increase is okay
        # loss_a has to decline
        self.model_zero_grad(names)
        # get name of the model parameters
        names = self.get_model_names(names)
        # backward loss_a
        self.detect_anomaly(loss_b)
        loss_b.backward(retain_graph=True)
        # normalize gradient
        b_grads = []
        for name in names:
            for p in self._models[name].parameters():
                b_grads.append(p.grad.clone())

        # optimizer don't step
        for name in names:
            self._optims[name].zero_grad()

        # backward loss_a
        self.detect_anomaly(loss_a)
        loss_a.backward()
        for name in names:
            for p, b_grad in zip(self._models[name].parameters(), b_grads):
                # calculate cosine distance
                b_grad_norm = b_grad / torch.linalg.norm(b_grad)
                a_grad = p.grad.clone()
                a_grad_norm = a_grad / torch.linalg.norm(a_grad)

                if torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten()) < 0:
                    p.grad = a_grad - lambda_ * torch.dot(
                        a_grad.flatten(), b_grad_norm.flatten()
                    ) * b_grad_norm

        # optimizer
        for name in names:
            self._optims[name].step()


class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname, num_classes=self.num_classes)
        self.best_result = -np.inf

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (except self.dm).
        """
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader_base = dm.val_base_loader  # optional, can be None
        self.test_loader_base = dm.test_base_loader
        self.val_loader_new = dm.val_new_loader  # optional, can be None
        self.test_loader_new = dm.test_new_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.ood_loaders = dm.ood_loaders
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print("# params: {:,}".format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(
                f"Detected {device_count} GPUs. Wrap the model with nn.DataParallel"
            )
            self.model = nn.DataParallel(self.model)

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        if self.cfg.model_dir and self.cfg.load_epoch:
            self.load_model(self.cfg.model_dir, epoch=self.cfg.load_epoch)
            self.start_epoch = self.cfg.load_epoch
        else:
            self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard", f"{self.cfg.DATASET.SUBSAMPLE_CLASSES}")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        '''if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    model_name="model-best.pth.tar"
                )'''
        if self.cfg.TEST.DO_VAL:
            self.test(split="val")
        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def output_test(self, split=None):
        """testing pipline, which could also output the results."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        output_file = osp.join(self.cfg.OUTPUT_DIR, 'output.json')
        res_json = {}

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            img_path = batch['impath']
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)
            for i in range(len(img_path)):
                res_json[img_path[i]] = {
                    'predict': output[i].cpu().numpy().tolist(),
                    'gt': label[i].cpu().numpy().tolist()
                }
        with open(output_file, 'w') as f:
            json.dump(res_json, f)
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def ood_score(self, global_logits=None, local_logits=None, method='energy', coeff=1., empty_cls=False):
        global_logits = global_logits / 100.0
        if 'mcm' in method:
            smax_global = F.softmax(global_logits/self.cfg.T, dim=-1)
            smax_global = (smax_global[:,:-1] - coeff*smax_global[:,-1:]) if empty_cls else smax_global
            score = -np.max(smax_global.detach().cpu().numpy(), axis=1)
            if method == 'glmcm':
                local_logits_ = local_logits / 100.0
                smax_local = F.softmax(local_logits_/self.cfg.T, dim=-1).detach().cpu().numpy()
                local_score = -np.max(smax_local, axis=(1, 2))
                score += local_score
        elif 'energy' in method:
            score = -(self.cfg.T*torch.logsumexp(global_logits / self.cfg.T, dim=-1)).detach().cpu().numpy()
        elif 'max_logit' in method:
            score = -(global_logits.max(-1)[0]).detach().cpu().numpy()
            
        return score
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        if self._writer is None:
            writer_dir = osp.join(self.output_dir, "tensorboard", f"{self.cfg.DATASET.SUBSAMPLE_CLASSES}")
            mkdir_if_missing(writer_dir)
            self.init_writer(writer_dir)

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader_base is not None and self.val_loader_new is not None:
            data_loader_base = self.val_loader_base
            data_loader_new = self.val_loader_new
        else:
            split = "test"  # in case val_loader is None
            data_loader_base = self.test_loader_base
            data_loader_new = self.test_loader_new

        if self.cfg.empty_cls_prompt:
            tfm_test = build_transform(self.cfg, is_train=False)
            train_data_loader = build_data_loader(
            self.cfg,
            sampler_type=self.cfg.DATALOADER.TEST.SAMPLER,
            data_source=self.train_loader_x.dataset.data_source,
            batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=None
            )
            train_base = []
            print('Calculating coefficient from training data')
            for batch in tqdm(train_data_loader): 
                self.set_model_mode("eval")
                with torch.no_grad():
                    input, label = self.parse_batch_test(batch)
                    output = self.model_inference(input)
                train_base.append(output)
            train_base = torch.cat(train_base,dim=0)

        print(f"Evaluate on the *{split}* set")
        loader = data_loader_base if self.cfg.DATASET.SUBSAMPLE_CLASSES == 'base' else data_loader_new
        self.evaluator.reset()
        in_outputs = []
        for batch in tqdm(loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            in_outputs.append(output)  
            self.evaluator.process(output, label) #if self.cfg.DATASET.SUBSAMPLE_CLASSES == subsample else None
            
        results = self.evaluator.evaluate() #if  == subsample else None
        for k, v in results.items():
            if not 'bin' in k:
                tag = f"{split}_{self.cfg.DATASET.SUBSAMPLE_CLASSES}/{k}"
                self.write_scalar(tag, v, self.epoch)


        if self.cfg.DATASET.SUBSAMPLE_CLASSES != 'new' and not self.cfg.no_eval_near_ood:
            print('Near-OOD Eval')
            near_ood_outputs = []
            for batch in tqdm(data_loader_new):
                input, label = self.parse_batch_test(batch)
                near_ood_outputs.append(self.model_inference(input))
            
            for ood_method in self.cfg.ood_method:
                if self.cfg.empty_cls_prompt:
                    x = train_base[:,-1] if 'mcm' not in ood_method else (train_base/100.).type(torch.float32).softmax(dim=-1)[:,-1]
                    y = torch.as_tensor(-self.ood_score(global_logits=train_base[:,:-1], method=ood_method, empty_cls=False), dtype=x.dtype, device=x.device)
                    coeff = ((((y-y.mean())*(x-x.mean())).sum())/(((x-x.mean())**2).sum())).type(train_base.dtype)
                else:
                    coeff = None

                in_score = []
                for output in tqdm(in_outputs, desc='ID '+ ood_method):
                    output = output[:,:-1]-coeff*output[:,-1:] if self.cfg.empty_cls_prompt and not 'mcm' in ood_method else output
                    score = self.ood_score(global_logits=output, method=ood_method, coeff=coeff, empty_cls=self.cfg.empty_cls_prompt)
                    in_score.append(score) 

                near_ood_score = []
                for output in tqdm(near_ood_outputs, desc='Near OOD '+ ood_method):
                    output = output[:,:-1]-coeff*output[:,-1:] if self.cfg.empty_cls_prompt and not 'mcm' in ood_method else output
                    score = self.ood_score(global_logits=output, method=ood_method, coeff=coeff, empty_cls=self.cfg.empty_cls_prompt)
                    near_ood_score.append(score) 

                in_score = np.concatenate(in_score, axis=0)
                near_ood_score = np.concatenate(near_ood_score, axis=0)

                min_length = min(in_score.shape[0],near_ood_score.shape[0])
                in_ind = data_loader_base.dataset.random_index
                near_ood_ind = data_loader_new.dataset.random_index
                assert(len(in_ind)==in_score.shape[0] and len(near_ood_ind)==near_ood_score.shape[0])
                auroc, aupr, fpr = get_measures(-in_score[in_ind[:min_length]], -near_ood_score[near_ood_ind[:min_length]])

                self.write_scalar(f'Near_OOD/Avg/AUROC-{ood_method}', auroc, self.epoch)
                self.write_scalar(f'Near_OOD/Avg/AUPR-{ood_method}', aupr, self.epoch)
                self.write_scalar(f'Near_OOD/Avg/FPR-{ood_method}', fpr, self.epoch)

                print('Near-OOD AUROC-{}: {:.3%}'.format(ood_method, auroc))

        if self.cfg.DATASET.SUBSAMPLE_CLASSES != 'new' and self.cfg.eval_ood:
            print('OOD eval')
            all_auroc, all_aupr, all_fpr = {ood_method: [] for ood_method in self.cfg.ood_method}, {ood_method: [] for ood_method in self.cfg.ood_method}, {ood_method: [] for ood_method in self.cfg.ood_method}
            for ood_name, ood_dataloader in self.ood_loaders.items():
                ood_outputs = []
                for batch in tqdm(ood_dataloader):
                    input, label = self.parse_batch_test(batch)
                    ood_outputs.append(self.model_inference(input))

                for ood_method in self.cfg.ood_method:
                    if self.cfg.empty_cls_prompt:
                        x = train_base[:,-1] if 'mcm' not in ood_method else (train_base/100.).type(torch.float32).softmax(dim=-1)[:,-1]
                        y = torch.as_tensor(-self.ood_score(global_logits=train_base[:,:-1], method=ood_method, empty_cls=False), dtype=x.dtype, device=x.device)
                        coeff = ((((y-y.mean())*(x-x.mean())).sum())/(((x-x.mean())**2).sum())).type(train_base.dtype)
                    else:
                        coeff = None

                    in_score = []
                    for output in tqdm(in_outputs, desc='ID '+ood_method):
                        output = output[:,:-1]-coeff*output[:,-1:] if self.cfg.empty_cls_prompt and not 'mcm' in ood_method else output
                        score = self.ood_score(global_logits=output, method=ood_method, coeff=coeff, empty_cls=self.cfg.empty_cls_prompt)
                        in_score.append(score) 

                    ood_score = []
                    for output in tqdm(ood_outputs, desc='OOD '+ ood_method):
                        output = output[:,:-1]-coeff*output[:,-1:] if self.cfg.empty_cls_prompt and not 'mcm' in ood_method else output
                        score = self.ood_score(global_logits=output, method=ood_method, coeff=coeff, empty_cls=self.cfg.empty_cls_prompt)
                        ood_score.append(score)
                        
                    in_score = np.concatenate(in_score, axis=0)
                    ood_score = np.concatenate(ood_score, axis=0)

                    auroc, aupr, fpr = get_measures(-in_score, -ood_score)
                    
                    self.write_scalar(f'OOD/{ood_name}/AUROC-{ood_method}', auroc, self.epoch)
                    self.write_scalar(f'OOD/{ood_name}/AUPR-{ood_method}', aupr, self.epoch)
                    self.write_scalar(f'OOD/{ood_name}/FPR-{ood_method}', fpr, self.epoch)

                    all_auroc[ood_method].append(auroc)
                    all_aupr[ood_method].append(aupr)
                    all_fpr[ood_method].append(fpr)

            for ood_method in self.cfg.ood_method:
                self.write_scalar(f'OOD/Avg/AUROC-{ood_method}', np.mean(all_auroc[ood_method]), self.epoch)
                self.write_scalar(f'OOD/Avg/AUPR-{ood_method}', np.mean(all_aupr[ood_method]), self.epoch)
                self.write_scalar(f'OOD/Avg/FPR-{ood_method}', np.mean(all_fpr[ood_method]), self.epoch)
                print('OOD AUROC: {:.3%}'.format(np.mean(all_auroc[ood_method])))


    def model_inference(self, input):
        return self.model(input)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]

class TrainerXU(SimpleTrainer):
    """A base trainer using both labeled and unlabeled data.

    In the context of domain adaptation, labeled and unlabeled data
    come from source and target domains respectively.

    When it comes to semi-supervised learning, all data comes from the
    same domain.
    """

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (
                self.batch_idx + 1
            ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "eta {eta}\t"
                    "{losses}\t"
                    "lr {lr:.6e}".format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr(),
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u


class TrainerX(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (
                self.batch_idx + 1
            ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "eta {eta}\t"
                    "{losses}\t"
                    "lr {lr:.6e}".format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr(),
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain