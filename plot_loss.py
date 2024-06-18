import pandas as pd
import matplotlib.pyplot as plt

import os

import argparse

from os.path import join as ospj
from parser import dataset_argparse, plot_loss_argparse


def plot_loss(csv_files, labels, columns, plot_name, save_plot=True, show_plot=False, combine=False):
    plt.figure(figsize=(10, 5))

    for csv_file, label in zip(csv_files, labels):
        data = pd.read_csv(csv_file)
        epochs = data['epoch']

        for col in columns:
            losses = data[col]
            plt.plot(epochs, losses, label=f'{label} {col}', markersize=3, linewidth=1)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss across Epochs')
    plt.legend()
    plt.grid(True)

    if save_plot:
        if combine:
            # Save combined plot in the parent directory to the current saving folder
            save_path = ospj(os.path.commonpath(csv_files), f'combined_{plot_name}.png')
            plt.savefig(save_path)
            print(f'Combined plot saved at: {save_path}')
        else:
            # Save individual plots
            for csv_file in csv_files:
                save_path = ospj(os.path.dirname(csv_file), f'{plot_name}.png')
                plt.savefig(save_path)
                print(f'Plot saved at: {save_path}')
                plt.clf()  # Clear figure for the next plot

    if show_plot:
        plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser = dataset_argparse(parser)
    parser = plot_loss_argparse(parser)

    args = parser.parse_args()

    save_path = ospj(args.save_dir, args.save_name, args.split_name)
    csv_files = []
    labels = []

    for num in args.nums:
        model_path = ospj(save_path, f'{num}', args.csv_file_name)
        
        if os.path.isfile(model_path):
            csv_files.append(model_path)
            labels.append(f'Model {num}')
        else:
            print(f'File not found: {model_path}')

    if args.combine:
        plot_loss(csv_files, labels, args.columns, args.plot_name, args.save_plot, args.show_plot, combine=True)
    else:
        for csv_file, label in zip(csv_files, labels):
            plot_loss([csv_file], [label], args.columns, args.plot_name, args.save_plot, args.show_plot)


if __name__ == '__main__':
    main()