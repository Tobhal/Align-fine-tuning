import argparse
from os.path import join as ospj

def plot_loss_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    plot_loss_argparse = parser.add_argument_group('Plot loss arguments')

    plot_loss_argparse.add_argument('--nums', required=True, nargs='+', type=int, help='A list of model numbers to process')
    plot_loss_argparse.add_argument('--save_plot', action='store_true', default=False, help='Save the plot to a file')
    plot_loss_argparse.add_argument('--show_plot', action='store_true', default=False, help='Show the plot on screen')
    plot_loss_argparse.add_argument('--csv_file_name', default='metrics.csv', help='Name of the CSV file containing loss values')
    plot_loss_argparse.add_argument('--combine', action='store_true', help='Plot all data on the same plot')
    plot_loss_argparse.add_argument('--columns', nargs='+', default=['train_loss', 'val_loss'], help='Columns to plot from the CSV file')
    plot_loss_argparse.add_argument('--plot_name', type=str, default='loss_plot')

    return parser