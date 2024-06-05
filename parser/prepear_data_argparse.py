import argparse
from os.path import join as ospj
from flags import DATA_FOLDER

def prepear_data_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    prepear_data_parser = parser.add_argument_group('Prepear data arguments')

    prepear_data_parser.add_argument('--original_data', type=str, default=ospj(DATA_FOLDER, 'BengaliWords', 'BengaliWords_CroppedVersion_Folds'), help='Path to the data directory')
    prepear_data_parser.add_argument('--fold', type=int, default=0, help='Fold number')

    prepear_data_parser.add_argument('--output_dir', type=str, default=ospj(DATA_FOLDER, 'BengaliWords', 'BengaliWords_CroppedVersion_Folds'), help='Path to the output directory')
    prepear_data_parser.add_argument('-n', '--output_name', type=str, default='t', help='Added name to the output folder')

    prepear_data_parser.add_argument('-a', '--augmented', action='store_true', help='Use augmented data')
    prepear_data_parser.add_argument('--augmentations_per_image', type=int, default=20, help='Number of augmentations per image')
    
    prepear_data_parser.add_argument('-s', '--split', action='store_true', help='Split the data')

    prepear_data_parser.add_argument('--split_ratio', type=float, default=0.2, help='Train-test split ratio')

    prepear_data_parser.add_argument('--down_scale_factor', type=float, default=0, help='Down scale factor')

    return parser