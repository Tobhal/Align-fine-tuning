#  Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv

#Local imports
from data import dataset_bengali as dset
from models.common import Evaluator
from flags import parser, DATA_FOLDER, device

from parser.phosc_net_argparse import phosc_net_argparse

from utils.utils import save_args, load_args
from utils.config_model import configure_model
from utils.dbe import dbe

from utils.combined_data_loader import CombinedLoader

# PHOSC utils
from modules.utils import set_phos_version, set_phoc_version
from modules import models, residualmodels
from timm import create_model

# CLIP
import clip

best_auc = 0
best_hm = 0
compose_switch = True

def main():
    # Get arguments and start logging
    p = phosc_net_argparse(parser)

    args = p.parse_args()
    load_args(args.config, args)
    # logpath = os.path.join(args.cv_dir, args.name, f'lr{args.lr:.1e}|lrg{args.lrg:.1e}|wd{args.wd:.1e}|cosine{args.cosine_scale}|aug{"t" if args.augmented else "f"}|nlayers{args.nlayers}')
    logpath = os.path.join(args.cv_dir, args.name, f'testing-no_attr_embed')

    os.makedirs(logpath, exist_ok=True)
    save_args(args, logpath, args.config)
    writer = SummaryWriter(log_dir=logpath, flush_secs = 30)

    # Define phosc model
    phosc_model = create_model(
        model_name=args.model_name,
        phos_size=args.phos_size,
        phoc_size=args.phoc_size,
        phos_layers=args.phos_layers,
        phoc_layers=args.phoc_layers,
        dropout=args.dropout
    ).to(device)

    args.phosc_model = phosc_model

    # Sett phos and phoc language
    set_phos_version(args.phosc_version)
    set_phoc_version(args.phosc_version)

    # Define CLIP model
    model_save_path = ospj('models', 'trained_clip', 'Fold0_use_50', 'bengali_word', '1', 'best.pt')
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    state_dict = torch.load(model_save_path, map_location=device)
    clip_model.load_state_dict(state_dict)

    args.clip_model = clip_model
    args.clip_preprocess = clip_preprocess

    # Get dataset
    train_set = dset.CompositionDataset(
        root=ospj(DATA_FOLDER, args.data_dir),
        phase='train',
        split=args.splitname,
        model=args.image_extractor,
        num_negs=args.num_negs,
        pair_dropout=args.pair_dropout,
        update_features = args.update_features,
        train_only=args.train_only,
        open_world=args.open_world,
        augmented=args.augmented,
        phosc_model=phosc_model,
        clip_model=clip_model
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    test_set = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='test',
        split=args.splitname,
        model =args.image_extractor,
        subset=args.subset,
        update_features = args.update_features,
        open_world=args.open_world,
        augmented=args.augmented,
        phosc_model=phosc_model,
        clip_model=clip_model
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, train_set)

    args.extractor = image_extractor

    train = train_normal

    evaluator_val = Evaluator(test_set, model)

    """
    # Function to process data from a loader and get embeddings
    def get_embeddings(loader, model):
        embeddings = []

        for i, batch in enumerate(loader):
            obj = batch[2].to(device)  # Move obj to the correct device
            obj_embedding = model.obj_embedder(obj)  # Pass obj as an argument
            embeddings.append(obj_embedding)

        return torch.vstack(embeddings)

    # Get embeddings for train and test sets
    train_embeddings = get_embeddings(trainloader, model)
    test_embeddings = get_embeddings(testloader, model)

    def calculate_cos_angle_matrix(vector_1, vector_2):
        n = len(vector_1)
        cos_angle_matrix = torch.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # Calculate the dot product of the two vectors
                dot_product = torch.dot(vector_1[i], vector_2[j])

                # Calculate the magnitudes of the vectors
                magnitude_a = torch.norm(vector_1[i])
                magnitude_b = torch.norm(vector_2[j])

                # Calculate the cosine of the angle
                cos_theta = dot_product / (magnitude_a * magnitude_b)

                # Ensure the cosine value is within the valid range [-1, 1] for arccos
                cos_theta = torch.clamp(cos_theta, -1, 1)

                # Assign the cosine value to the matrix
                cos_angle_matrix[i, j] = cos_theta

        return cos_angle_matrix
    
    def save_angle(angle, filename_part):
        # Convert the matrix to a NumPy array
        cos_angle_np = angle.cpu().detach().numpy()

        # Save the matrix to a file as raw text
        filename = f"cos_angle_all_{filename_part}.csv"
        file_path = ospj(DATA_FOLDER, args.data_dir, args.splitname, filename)

        with open(file_path, 'w') as file:
            for row in cos_angle_np:
                # Convert each row to a string and write it to the file
                row_str = ','.join(map(str, row))
                file.write(row_str + '\n')

    cos = calculate_cos_angle_matrix(test_embeddings, test_embeddings)
 
    min = torch.min(cos).item()
    mean = torch.mean(cos)
    std = torch.std(cos)

    dbe(min, mean, std)

    save_angle(
        calculate_cos_angle_matrix(test_embeddings, test_embeddings), 
        'test-test'
    )
    save_angle(
        calculate_cos_angle_matrix(test_embeddings, train_embeddings), 
        'test-train'
    )
    save_angle(
        calculate_cos_angle_matrix(train_embeddings, test_embeddings), 
        'train-test'
    )
    save_angle(
        calculate_cos_angle_matrix(train_embeddings, train_embeddings), 
        'train-train'
    )
    """
    print(model)

    start_epoch = 0
    # Load checkpoint
    if args.load is not None:
        checkpoint = torch.load(args.load)

        if image_extractor:
            try:
                image_extractor.load_state_dict(checkpoint['image_extractor'])

                if args.freeze_features:
                    print('Freezing image extractor')
                    image_extractor.eval()

                    for param in image_extractor.parameters():
                        param.requires_grad = False

            except:
                print('No Image extractor in checkpoint')

        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', args.load)
    
    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc='Current epoch'):
        train(epoch, image_extractor, model, train_loader, optimizer, writer)
        # train(epoch, image_extractor, model, train_loader, optimizer, writer)

        if model.is_open and args.model == 'compcos' and ((epoch + 1) % args.update_feasibility_every) == 0:
            print('Updating feasibility scores')
            model.update_feasibility(epoch+1.)

        if epoch % args.eval_val_every == 0:
            with torch.no_grad(): # todo: might not be needed
                test(epoch, image_extractor, model, test_loader, evaluator_val, writer, args, logpath)
                # test(epoch, image_extractor, model, test_loader, evaluator_val, writer, args, logpath)
                
    print('Best AUC achieved is ', best_auc)
    print('Best HM achieved is ', best_hm)


def train_normal(epoch, image_extractor, model, train_loader, optimizer, writer):
    '''
    Runs training for an epoch
    '''
    if image_extractor:
        image_extractor.train()

    model.train() # Let's switch to training

    train_loss = 0.0 
    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc = 'Training'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0] = image_extractor(data[0])

        loss, _ = model(data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()

    train_loss = train_loss/len(train_loader)

    writer.add_scalar('Loss/train_total', train_loss, epoch)
    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))


def test(epoch, image_extractor, model, test_loader, evaluator, writer, args, logpath):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm

    def save_checkpoint(filename):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'AUC': stats['AUC']
        }

        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()
        
        torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename)))

    if image_extractor:
        image_extractor.eval()

    model.eval()

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

    for idx, data in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0] = image_extractor(data[0])

        _, predictions = model(data)

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    if args.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to('cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    if args.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))]
            )
    else:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.stack(
                [all_pred[i][k] for i in range(len(all_pred))]
            )

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        writer.add_scalar(key, stats[key], epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    result = result + args.name
    print(f'Test Epoch: {epoch}')
    print(result)

    if epoch > 0 and epoch % args.save_every == 0:
        save_checkpoint(epoch)

    if stats['AUC'] > best_auc:
        improvement = stats['AUC'] - best_auc  # Calculate improvement
        best_auc = stats['AUC']  # Update best_auc
        print(f'New best AUC {best_auc}, improved by: {improvement:.4f}')  # Print improvement
        save_checkpoint('best_auc')

    if stats['best_hm'] > best_hm:
        improvement = stats['best_hm'] - best_hm  # Calculate improvement
        best_hm = stats['best_hm']  # Update best_hm
        print(f'New best HM {best_hm}, improved by: {improvement:.4f}')  # Print improvement
        save_checkpoint('best_hm')

    # Logs
    with open(ospj(logpath, 'logs.csv'), 'a') as f:
        w = csv.DictWriter(f, stats.keys())

        if epoch == 0:
            w.writeheader()

        w.writerow(stats)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Best AUC achieved is ', best_auc)
        print('Best HM achieved is ', best_hm)