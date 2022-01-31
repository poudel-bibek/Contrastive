from tqdm import tqdm
import argparse

from pipeline import Pipeline
from utils.stats_utils import generate_linegraph_augvsrecon, calc_average_stats
from simCLR_resnet_50 import ProjectionHead, PreModel, LinearLayer, Identity


def main(args):
    pl = Pipeline(args)
    pl.train()

    aug_list = get_aug_list('./aug_list.txt')     

    # Testing on the 3 benchmark datasets: clean, combined, & unseen
    for i in tqdm(range(len(aug_list))):
        print(f"\n{i+1} {aug_list[i]}")

        pl = Pipeline(args, aug_list[i], "test")
        pl.test(aug_list[i])

    generate_linegraph_augvsrecon('./logs/results_standard.txt', './logs/results_robust.txt')
    calc_average_stats('./logs/results_standard.txt', './logs/results_robust.txt','./logs/train_log.txt')


def get_aug_list(aug_list_path):
    aug_list = []

    with open(aug_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            aug_list.append(line) 
    
    return aug_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", default=8, help="Size of training batch")
    parser.add_argument("--lr", default=1e-4, help="Learning rate")
    parser.add_argument("--data_dir", default="data/", help="Data directory")
    parser.add_argument("--train_epochs", default=50, help="Number of training epochs")
    parser.add_argument("--model_dir", default='./saved_models/autoencoder.pt', help="Path to saved models")

    main(parser.parse_args())