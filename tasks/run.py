import os

os.environ["OMP_NUM_THREADS"] = "1"

from utils.commons.hparams import hparams, set_hparams
import importlib


def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])     # the package name before the class name
    cls_name = hparams["task_cls"].split(".")[-1]           # the class name
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    task_cls.start()    # call the original BaseTask.start()


if __name__ == '__main__':
    set_hparams()
    run_task()
