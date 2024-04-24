import torch
import matplotlib.pyplot as plt

if __name__=='__main__':
    state_dict = torch.load('../pretrained_segformer/checkpoint.pth', map_location=lambda storage, loc: storage)
    plt.plot(state_dict['history']['train']['loss'])
    plt.show()
    plt.plot(state_dict['history']['train']['acc'])
    plt.show()