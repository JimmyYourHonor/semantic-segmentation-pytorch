from abc import ABC, abstractmethod

class Log(ABC):

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        pass

    def before_backward(self, **kwargs):
        pass

    def before_optim(self):
        pass

    def after_optim(self):
        pass

    def on_valid_epoch_start(self):
        pass

    def on_valid_epoch_end(self):
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