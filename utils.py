from collections import OrderedDict
from collections import namedtuple
from itertools import product

import json
import pandas as pd
from IPython.display import display, clear_output
import time

from torch.utils.tensorboard import SummaryWriter

import torch
import torchvision


class Epoch():
    def __init__(self, loader):
        self.loader = loader

        self.start_time = time.time()
        self.loss = 0
        self.num_correct = 0

    def track_loss(self, loss):
        self.loss += loss.item()

    def track_num_correct(self, preds, labels):
        self.num_correct += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def end(self):
        self.duration = time.time() - self.start_time
        self.loss = self.loss / len(self.loader.dataset)
        self.accuracy = self.num_correct / len(self.loader.dataset)

    def to_dict(self):
        d = {
            'duration': self.duration,
            'loss': self.loss,
            'nu_correct': self.num_correct,
            'accuracy': self.accuracy
        }
        return d


class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


class RunManager():
    def __init__(self, file_name):
        self.file_name = file_name

        self.epoch_count = 0

        self.params = None
        self.run_count = 0
        self.data = []
        self.start_time = None

        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader):
        self.start_time = time.time()
        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def write_epoch(self, epoch):
        run_duration = time.time() - self.start_time

        self.tb.add_scalar('Loss', epoch['loss'], self.epoch_count)
        self.tb.add_scalar('Accuracy', epoch['accuracy'], self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['loss'] = epoch['loss']
        results['accuracy'] = epoch['accuracy']
        results['epoch duration'] = epoch['duration']
        results['run duration'] = run_duration

        self.epoch_count += 1

        for k, v in self.run_params._asdict().items():
            results[k] = v

        self.data.append(results)
        df = pd.DataFrame.from_dict(self.data, orient='columns')

        clear_output(wait=True)
        display(df)

    def save(self):
        pd.DataFrame.from_dict(
            self.data, orient='columns'
        ).to_csv(f'{self.file_name}.csv')

        with open(f'{self.file_name}.json', 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.save()
