import argparse
from voc import *

def parse_args():
    parser = argparse.ArgumentParser(description="VOC")
    parser.add_argument('--phase', type=str, default='rp', help='train/test/rp/indel/c_sen')
    parser.add_argument('--model_name', type=str, default='ResNet', help='GoogLeNet / ResNet / VGG')
    parser.add_argument('--dataset', type=str, default='VOC', help='VOC')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='total training epoch')
    parser.add_argument('--lr', type=float, default=0.0001, help='total training epoch')

    parser.add_argument('--resume', type=bool, default=False, help='True or Fasle for resume training from last point')
    parser.add_argument('--device', type=str, default='cuda:0', help='select cuda device')
    parser.add_argument('--print_freq', type=int, default=50, help='the number of print frequency')
    # Change data directory for train network from scratch
    parser.add_argument('--data_dir', type=str, default='/mnt/hard1/jh_datasets/VOC2007/', help='dataset dir')
    # Change data directory for generating attribution maps
    parser.add_argument('--xai_data', type=str, default='/mnt/hard1/jh_datasets/VOC2007/VOCdevkit/VOC2007', help='xai dataset dir')
    parser.add_argument('--log_dir', type=str, default='logs', help='directory of training logs')
    parser.add_argument('--xai_dir', type=str, default='xai', help='directory of explainable')
    parser.add_argument('--result_dir', type=str, default='results', help='directory of generated images at test')
    return parser.parse_args()

def main():
    args = parse_args()
    if args is None:
        exit()

    check_dir(args.log_dir)
    check_dir(args.xai_dir)
    check_dir(args.result_dir)

    # open session
    model = VOC(args)

    # build graph
    model.build_model()

    if args.phase == 'train':
        print(" [*] Training started!")
        model.train()
        print(" [*] Training finished!")

    elif args.phase == 'test':
        print(" [*] Test started!")
        model.test()
        print(" [*] Test finished!")

    elif args.phase == 'rp':
        print(" [*] Robust Perturbation started!")
        model.robust_perturbation()
        print(" [*] Robust Perturbation finished!")

    elif args.phase == 'indel':
        print(" [*] Insertion and Deletion for %s started!" %(args.method))
        model.indel_game()
        print(" [*] Insertion and Deletion for %s finished!" %(args.method))
    
    elif args.phase == 'energy':
        print(" [*] Energy-based Pointing Game for %s started!" %(args.method))
        model.energy()
        print(" [*] Energy-based Pointing Game for %s finished!" %(args.method))
    
    elif args.phase == 'c_sen':
        print(" [*] Class Sensitivity for %s started!" %(args.method))
        model.c_sen()
        print(" [*] Class Sensitivity for %s finished!" %(args.method))

if __name__ == "__main__":
    main()
