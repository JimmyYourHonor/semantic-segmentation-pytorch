import torch

if __name__=='__main__':
    state_dict = torch.load('ckpt/ade20k-segformer-b0/checkpoint_epoch_1.pth')
    print(state_dict['seed'])
    print(torch.get_rng_state())
    torch.set_rng_state(state_dict['seed'])
    print(torch.get_rng_state())