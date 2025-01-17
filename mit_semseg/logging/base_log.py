from abc import ABC, abstractmethod

class Log(ABC):

    def on_train_epoch_start(self, model):
        pass

    def on_train_epoch_end(self, model):
        pass

    def before_backward(self, input, loss, target, epoch):
        pass

    def before_optim(self, model):
        pass

    def after_optim(self, model):
        pass

    def on_valid_epoch_start(self, model):
        pass

    def on_valid_epoch_end(self, model):
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint):
        pass
    
    @abstractmethod
    def save_results(self, cfg):
        pass