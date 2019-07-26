import torch
from abc import ABCMeta
from torch import nn as nn


class BaseNetMCD(nn.Module, metaclass=ABCMeta):
    STATE_KEY_GEN = "generator"
    STATE_KEY_CLS_F1 = "classifier_f1"
    STATE_KEY_CLS_F2 = "classifier_f2"

    SUPPORT = [
        STATE_KEY_GEN,
        STATE_KEY_CLS_F1,
        STATE_KEY_CLS_F2
    ]

    def __init__(self, ckpt: str = None):
        super().__init__()
        self.generator: nn.Module
        self.classifier_f1: nn.Module
        self.classifier_f2: nn.Module

    def load(self, path):
        state_dict = torch.load(str(path))
        try:
            self.generator.load_state_dict(state_dict[self.STATE_KEY_GEN])

            self.classifier_f1.load_state_dict(state_dict[self.STATE_KEY_CLS_F1])
            self.classifier_f2.load_state_dict(state_dict[self.STATE_KEY_CLS_F2])
        except KeyError:
            keys = list(state_dict.keys())
            msg = str(
                "KeyErrored skip: \n"
                f"\tReceive: {keys}\n"
                f"\tExpect : {self.SUPPORT}"
            )
            print(msg)
        return self

    def save(self, path: str) -> None:
        print("*** saving model ...", end="")
        state_dict = {
            self.STATE_KEY_GEN: self.generator.state_dict(),
            self.STATE_KEY_CLS_F1: self.classifier_f1.state_dict(),
            self.STATE_KEY_CLS_F2: self.classifier_f2.state_dict(),
        }
        torch.save(state_dict, str(path))
        print("DONE")

    def parameters(self, recurse=True):
        gen_params = self.generator.parameters(recurse=recurse)

        f1_cls_params = self.classifier_f1.parameters(recurse=recurse)
        f2_cls_params = self.classifier_f2.parameters(recurse=recurse)

        return gen_params, f1_cls_params, f2_cls_params

    def train_all(self):
        self.generator.train(True)
        self.classifier_f1.train(True)
        self.classifier_f2.train(True)

    def train_classifier(self):
        self.generator.train(False)
        self.classifier_f1.train(True)
        self.classifier_f2.train(True)

    def train_generator(self):
        self.generator.train(True)
        self.classifier_f1.train(False)
        self.classifier_f2.train(False)
