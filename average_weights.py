import os
import argparse
import torch


def get_checkpoints(args):
    checkpoint_dir = args.log_dir + '/' + args.model_name + f'/fold_{args.fold}' + '/checkpoint'
    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir) if checkpoint.endswith('.pth')]

    checkpoints = [os.path.join(checkpoint_dir, ckpt) for ckpt in sorted(checkpoints)]
    checkpoints = checkpoints[-args.num_checkpoint:]
    return checkpoints

def run_swa(args):

    save_dir = args.log_dir + '/' + args.model_name + f'/fold_{args.fold}' + '/checkpoint'
    ckpts = get_checkpoints(args)
    num_snapshot = len(ckpts)
    save_name = f'model_swa_no_bn_{num_snapshot}.pth'

    swa_state_dict = torch.load(ckpts[0])['state_dict']
    
    for k,v in swa_state_dict.items():
        swa_state_dict[k] = torch.zeros_like(v)

    for ckpt in ckpts[1:]:
        print(ckpt)
        state_dict = torch.load(ckpt)['state_dict']
        for k,v in state_dict.items():
            swa_state_dict[k] += v

    for k,v in swa_state_dict.items():
        swa_state_dict[k] /= num_snapshot

    weights_dict = {'state_dict': swa_state_dict}
    torch.save(weights_dict, save_dir +f'/{save_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle SIIM Competition')
    parser.add_argument('--gpu', type=int, default=0, 
                    help='Choose GPU to use. This only support single GPU')
    parser.add_argument('--log_dir', type=str, default='runs', 
                    help='logging directory')               
    parser.add_argument('--fold', type=int, default=0,
                    help='which fold to use for training')
    parser.add_argument('--model_name', type=str, default='FPN_effb4',
                    help='model_name as exp_name')
    parser.add_argument('--encoder_name', type=str, default='resnet34', 
                    help='which encode to use for model')
    parser.add_argument('--decoder_name', type=str, default='Unet',
                    help='type of decoder to use for model')
    parser.add_argument('--num_checkpoint', type=int, default=5,
                    help='how many snapshots to use for swa')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']= f'{args.gpu}'

    # run swa
    run_swa(args)
