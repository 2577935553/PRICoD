import os
import random
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import tqdm
import nibabel as nib
from tensorboardX import SummaryWriter

from SegModel import ProjectUNet
from utilities import losses
from utilities.val_2D import test_single_volume, get_image_list, crop_image
from utilities.MyDataSet import SemiSegDataset
from utilities.Load_Data_v2 import augment_data_batch
from modules import PixelRegionInstanceContrast, ClassConditionedRefinement, urc_loss

def create_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=0):
    """
    Linear warmup followed by cosine annealing.

    Falls back to plain cosine annealing when there is no room for a warmup phase
    (e.g. very short schedules used by the smoke test), which keeps LinearLR and
    SequentialLR from receiving degenerate ``total_iters=0`` / ``milestones=[0]``.
    """
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(int(warmup_epochs), 0)
    # need at least one warmup step and one cosine step for the two-phase schedule
    if warmup_epochs < 1 or warmup_epochs >= total_epochs:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,  # Start at 1% of base LR
        end_factor=1.0,     # End at 100% of base LR
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=eta_min
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)


def sigmoid_rampup(current, rampup_length):
    """
    Exponential rampup from https://arxiv.org/abs/1610.02242
    Args:
        current: current epoch index
        :param current:
        :param rampup_length:
    """
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="PRICoD training / testing")
    # Data (all paths default to ./data under the repo root)
    parser.add_argument("--data_dir", type=str, default='./data/ACDC/training',
                        help="Root directory of the training/validation volumes")
    parser.add_argument("--train_data_csv", type=str, default='./data_dir/train_ACDC.csv',
                        help="CSV listing the training subjects")
    parser.add_argument("--valid_data_csv", type=str, default='./data_dir/valid_ACDC.csv',
                        help="CSV listing the validation subjects")
    parser.add_argument("--test_data_dir", type=str, default='./data/ACDC/testing',
                        help="Root directory of the test volumes")
    parser.add_argument("--test_data_list", type=str, default='./data_dir/test_subj_ACDC.csv',
                        help="CSV listing the test subjects")
    parser.add_argument("--exp_dir", type=str, default='./outputs/ACDC_10',
                        help="Experiment directory for checkpoints, logs and predictions")

    parser.add_argument("--mode", type=str, default='train', help="train | test")
    parser.add_argument("--label_ratio", type=float, default=0.1, help="Fraction of labeled subjects")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument('--lambda1', type=float, default=2.0, help="Weight of the URC consistency loss")
    parser.add_argument('--lambda2', type=float, default=0.01, help="Weight of the PRI contrastive loss")
    parser.add_argument('--device', type=int, default=0, help="GPU device id")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_iterations", type=int, default=30000, help="Total training iterations")
    parser.add_argument("--val_interval", type=int, default=200,
                        help="Run validation and save the best checkpoint every N iterations")
    parser.add_argument("--learning_rate", type=float, default=0.02, help="Base learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="(unused placeholder)")
    parser.add_argument("--image_size", nargs='+', type=int, default=[224, 224], help="Input image size")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of output classes")

    # Module switches (the full PRICoD model enables --pri --crm --crm_of --ensemble)
    parser.add_argument("--pri", action='store_true', default=False,
                        help="Enable Pixel-Region-Instance contrast (PRI)")
    parser.add_argument("--crm", action='store_true', default=False,
                        help="Enable the CRM refinement module (Class-conditioned Refinement Module)")
    parser.add_argument("--crm_of", action='store_true', default=False,
                        help="Use the CRM-refined logits as the prediction at eval/inference")
    parser.add_argument("--ensemble", action='store_true', default=False,
                        help="Also keep the second network (Seg2) and average both at inference")

    return parser.parse_args()

def training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_torch(args.seed)
    g_new_worker = torch.Generator()
    g_new_worker.manual_seed(args.seed)
    
    exp_dir = args.exp_dir
    model_dir = exp_dir + '/model'
    training_graph = exp_dir + '/graph'
    train_csv = exp_dir + '/csv'
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = int(args.epochs)
    image_size = args.image_size
    num_classes = args.num_classes

    max_iterations = args.max_iterations
    mkdir(exp_dir)
    mkdir(model_dir)
    mkdir(training_graph)
    mkdir(train_csv)

    train_set = SemiSegDataset(args.data_dir, args.train_data_csv, args.valid_data_csv, image_size,
                               label_ratio=args.label_ratio, mode='train', random_seed=args.seed)
    valset = SemiSegDataset(args.data_dir, args.train_data_csv, args.valid_data_csv, image_size,
                            label_ratio=args.label_ratio, mode='valid', random_seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, drop_last=True,shuffle=True,generator=g_new_worker)

    val_loader = DataLoader(valset, batch_size=20, shuffle=False)

    model1 = ProjectUNet('resnet50', None, classes=num_classes, deep_stem=32).cuda()
    model2 = ProjectUNet('resnet50', None, classes=num_classes, deep_stem=32).cuda()

    crm = ClassConditionedRefinement().cuda()          # class-conditioned logit refinement
    pri_CL = PixelRegionInstanceContrast(feature_dim=64).cuda()  # pixel-region-instance contrast
    writer = SummaryWriter(training_graph)

    model1.train()
    model2.train()
    pri_CL.train()
    crm.train()

    contrastive_loss = pri_CL

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    iter_num = 0
    max_epoch = args.max_iterations // len(train_loader) + 1

    max_iterations = args.max_iterations
    iter_num_T = int(max_iterations * 0.3)
    best_performance1 = -1.0  # ensures the first validation always writes a checkpoint

    label_init_1=False
    label_init_2=False
    optimizer1 = None
    iterator = tqdm.tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:

        current_lr = optimizer1.param_groups[0]['lr'] if optimizer1 is not None else args.learning_rate * 0.04
        print("Epoch {}/{}, current lr {}".format(epoch + 1, max_epoch, current_lr))

        avloss_model1_suploss = 0.0
        avloss_model2_suploss = 0.0
        avloss_model1_unsuploss = 0.0
        avloss_model2_unsuploss = 0.0
        avloss_model1_contrastive_loss = 0.0
        avloss_model2_contrastive_loss = 0.0

        for step, (batch_labeled_data, batch_labeled_lab, batch_unlabeled_data) in enumerate(train_loader):
            if iter_num<iter_num_T and (not label_init_1):
                warmup_epochs=int(iter_num_T*0.1)
                total_epochs_step1=iter_num_T
                optimizer1 = optim.AdamW(model1.parameters(), lr=args.learning_rate*0.04, weight_decay=0.0001)
                optimizer2 = optim.AdamW(model2.parameters(), lr=args.learning_rate*0.04, weight_decay=0.0001)
                optimizer_pri= optim.AdamW(pri_CL.parameters(), lr=args.learning_rate*0.04, weight_decay=0.0001)
                optimizer_crm= optim.AdamW(crm.parameters(), lr=args.learning_rate*0.04, weight_decay=0.0001)
                scheduler1 = create_warmup_cosine_scheduler(optimizer1, warmup_epochs, total_epochs_step1)
                scheduler2 = create_warmup_cosine_scheduler(optimizer2, warmup_epochs, total_epochs_step1)
                scheduler_pri = create_warmup_cosine_scheduler(optimizer_pri, warmup_epochs, total_epochs_step1)
                scheduler_crm = create_warmup_cosine_scheduler(optimizer_crm, warmup_epochs, total_epochs_step1)
                label_init_1=True
            if iter_num>=iter_num_T and (not label_init_2):
                warmup_epochs=int((max_iterations-iter_num_T)*0.1)
                total_epochs_step2=max_iterations-iter_num_T
                optimizer1 = optim.SGD(model1.parameters(), lr=args.learning_rate,
                        momentum=0.9, weight_decay=0.0001)
                optimizer2 = optim.SGD(model2.parameters(), lr=args.learning_rate,
                                    momentum=0.9, weight_decay=0.0001)
                optimizer_pri= optim.SGD(pri_CL.parameters(), lr=args.learning_rate,
                                    momentum=0.9, weight_decay=0.0001)
                optimizer_crm= optim.SGD(crm.parameters(), lr=args.learning_rate,
                                    momentum=0.9, weight_decay=0.0001)
                scheduler1 = create_warmup_cosine_scheduler(optimizer1, warmup_epochs, total_epochs_step2)
                scheduler2 = create_warmup_cosine_scheduler(optimizer2, warmup_epochs, total_epochs_step2)
                scheduler_pri = create_warmup_cosine_scheduler(optimizer_pri, warmup_epochs, total_epochs_step2)
                scheduler_crm = create_warmup_cosine_scheduler(optimizer_crm, warmup_epochs, total_epochs_step2)
                label_init_2=True
            model1.train()
            model2.train()
            if args.crm:
                crm.train()
            aug_labeled_img, aug_labeled_lab = batch_labeled_data.numpy().squeeze(), batch_labeled_lab.numpy()
            aug_labeled_img, aug_labeled_lab = augment_data_batch(aug_labeled_img, aug_labeled_lab, shift=15, rotate=30,
                                                                  scale=0.2,flip=False)
            aug_labeled_img, aug_labeled_lab = \
                torch.Tensor(aug_labeled_img).unsqueeze(1), torch.Tensor(aug_labeled_lab).long()
            aug_unlabeled_img = batch_unlabeled_data.numpy().squeeze()
            aug_unlabeled_img, _ = augment_data_batch(aug_unlabeled_img, aug_unlabeled_img, shift=15, rotate=30,
                                                      scale=0.2,flip=False)
            aug_unlabeled_img = torch.Tensor(aug_unlabeled_img).unsqueeze(1)
            image_labeled = aug_labeled_img.to(device)
            label_labeled = aug_labeled_lab.to(device)
            image_unlabeled = aug_unlabeled_img.to(device)

            if not args.crm:
                out_l_1, proj_l_1 = model1(image_labeled)
                out_l_soft1 = F.softmax(out_l_1, dim=1)

                out_u_1, proj_u_1 = model1(image_unlabeled)
                out_u_soft1 = F.softmax(out_u_1, dim=1)

                out_l_2, proj_l_2 = model2(image_labeled)
                out_l_soft2 = F.softmax(out_l_2, dim=1)

                out_u_2, proj_u_2 = model2(image_unlabeled)
                out_u_soft2 = F.softmax(out_u_2, dim=1)

                loss1 = 0.5 * ce_loss(out_l_1, label_labeled) + dice_loss(
                out_l_soft1, label_labeled
                )
                loss2 = 0.5 * ce_loss(out_l_2, label_labeled) + dice_loss(
                out_l_soft2, label_labeled
            )
            else:
                out_l_1_o, proj_l_1 = model1(image_labeled)
                out_l_soft1_o = F.softmax(out_l_1_o, dim=1)
                out_l_1 = crm(out_l_1_o)

                out_u_1_o, proj_u_1 = model1(image_unlabeled)
                out_u_soft1_o = F.softmax(out_u_1_o, dim=1)
                out_u_1 = crm(out_u_1_o)
                out_l_soft1 = F.softmax(out_l_1, dim=1)
                out_u_soft1 = F.softmax(out_u_1, dim=1)

                out_l_2_o, proj_l_2 = model2(image_labeled)
                out_l_soft2_o = F.softmax(out_l_2_o, dim=1)
                out_l_2 = crm(out_l_2_o)

                out_u_2_o, proj_u_2 = model2(image_unlabeled)
                out_u_soft2_o = F.softmax(out_u_2_o, dim=1)
                out_u_2 = crm(out_u_2_o)
                out_l_soft2 = F.softmax(out_l_2, dim=1)
                out_u_soft2 = F.softmax(out_u_2, dim=1)

                loss1_o = 0.5 * ce_loss(out_l_1_o, label_labeled) + dice_loss(
                out_l_soft1_o, label_labeled
                )
                loss2_o = 0.5 * ce_loss(out_l_2_o, label_labeled) + dice_loss(
                    out_l_soft2_o, label_labeled
                )
                loss1 = 0.5 * ce_loss(out_l_1, label_labeled) + dice_loss(
                    out_l_soft1, label_labeled
                )+loss1_o
                loss2 = 0.5 * ce_loss(out_l_2, label_labeled) + dice_loss(
                    out_l_soft2, label_labeled
                )+loss2_o
                
            pseudo_out_l_1 = torch.argmax(out_l_soft1.detach(), dim=1)
            pseudo_out_u_1 = torch.argmax(out_u_soft1.detach(), dim=1)

            pseudo_out_l_2 = torch.argmax(out_l_soft2.detach(), dim=1)
            pseudo_out_u_2 = torch.argmax(out_u_soft2.detach(), dim=1)
            _, C, *spatial_size = proj_l_1.shape
            spatial_dim = len(spatial_size)

            # URC: uncertainty-rectified cross-network consistency on the (refined) logits
            pseudo_supervision_l_1 = urc_loss(out_l_1, out_l_2)
            pseudo_supervision_u_1 = urc_loss(out_u_1, out_u_2)
            pseudo_supervision_l_2 = urc_loss(out_l_2, out_l_1)
            pseudo_supervision_u_2 = urc_loss(out_u_2, out_u_1)

            # PRI: pixel-region-instance contrast across the two networks
            if args.pri:
                contrastive_loss_l_1 = contrastive_loss(proj_l_1, proj_l_2, proj_u_1, proj_u_2,
                                                        out_l_1, out_l_2, out_u_1, out_u_2)
                contrastive_loss_u_1 = contrastive_loss(proj_u_1, proj_u_2, proj_l_1, proj_l_2,
                                                        out_u_1, out_u_2, out_l_1, out_l_2)
                contrastive_loss_l_2 = contrastive_loss(proj_l_2, proj_l_1, proj_u_1, proj_u_2,
                                                        out_l_2, out_l_1, out_u_1, out_u_2)
                contrastive_loss_u_2 = contrastive_loss(proj_u_2, proj_u_1, proj_l_1, proj_l_2,
                                                        out_u_2, out_u_1, out_l_1, out_l_2)
            else:
                zero = torch.zeros((), device=image_labeled.device)
                contrastive_loss_l_1 = contrastive_loss_u_1 = zero
                contrastive_loss_l_2 = contrastive_loss_u_2 = zero

            tau = sigmoid_rampup(iter_num // 100*100, args.max_iterations)
            model1_loss = loss1 + lambda1 * (pseudo_supervision_l_1 + pseudo_supervision_u_1) \
                          + lambda2 * tau * (contrastive_loss_l_1 + contrastive_loss_u_1)
            model2_loss = loss2 + lambda1 * (pseudo_supervision_l_2 + pseudo_supervision_u_2) \
                          + lambda2 * tau * (contrastive_loss_l_2 + contrastive_loss_u_2)
                          
            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            if args.pri:
                optimizer_pri.zero_grad()
            if args.crm:
                optimizer_crm.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()
            if args.pri:
                optimizer_pri.step()
            if args.crm:
                optimizer_crm.step()

            iter_num += 1
            scheduler1.step()
            scheduler2.step()
            scheduler_pri.step()
            scheduler_crm.step()
            avloss_model1_suploss += loss1.item()
            avloss_model2_suploss += loss2.item()
            avloss_model1_unsuploss += pseudo_supervision_l_1.item() + pseudo_supervision_u_1.item()
            avloss_model2_unsuploss += pseudo_supervision_l_2.item() + pseudo_supervision_u_2.item()
            avloss_model1_contrastive_loss += contrastive_loss_l_1.item() + contrastive_loss_u_1.item()
            avloss_model2_contrastive_loss += contrastive_loss_l_2.item() + contrastive_loss_u_2.item()
            writer.add_scalar('lr', scheduler1.get_last_lr()[0], iter_num)
            writer.add_scalar('tau', tau, iter_num)
            writer.add_scalar('loss/model1loss', model1_loss.item(), iter_num)
            writer.add_scalar('loss/model2loss', model2_loss.item(), iter_num)
            writer.add_scalar('loss/model1_unsup_loss', pseudo_supervision_l_1.item() + pseudo_supervision_u_1.item(),
                              iter_num)
            writer.add_scalar('loss/model2_unsup_loss', pseudo_supervision_l_2.item() + pseudo_supervision_u_2.item(),
                              iter_num)
            writer.add_scalar('loss/model1_contrastive_loss', contrastive_loss_l_1.item() + contrastive_loss_u_1.item(),
                              iter_num)
            writer.add_scalar('loss/model2_contrastive_loss', contrastive_loss_l_2.item() + contrastive_loss_u_2.item(),
                              iter_num)

            iterator.desc = "train Epoch[{}/{}] sup_loss:{:.3f}, unsup_loss:{:.3f}, contrastive_loss:{:.3f}".format(
                epoch, epoch + 1, loss1.item(), pseudo_supervision_l_1.item() + pseudo_supervision_u_1.item(),
                       contrastive_loss_l_1.item() + contrastive_loss_u_1.item()
            )

            if iter_num % 50 == 0:
                image = image_labeled[1, 0:1, :, :]
                writer.add_image('train/Image', torch.Tensor(image.cpu()), iter_num)
                outputs = torch.argmax(F.softmax(out_l_1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_pred', outputs[1, ...].detach().cpu().numpy().astype(float) * 50,
                                 iter_num)
                unlab_image = image_unlabeled[1, 0:1, :, :]
                writer.add_image('train/unlab_image', torch.Tensor(unlab_image.cpu()), iter_num)
                unlab_outs = torch.argmax(F.softmax(out_u_1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_unlab_pred',
                                 unlab_outs[1, ...].detach().cpu().numpy().astype(float) * 50, iter_num)
                gt = label_labeled[1, :, :].unsqueeze(0)
                writer.add_image('train/GT', torch.FloatTensor(gt.detach().cpu().numpy().astype(float)), iter_num)

            if iter_num > 0 and iter_num % args.val_interval == 0:
                model1.eval()
                metric_list = 0.0
                for sample_batch in val_loader:
                    metric_i = test_single_volume(
                        sample_batch[0], sample_batch[1],
                        model1, classes=num_classes, crm_md=crm, crm_of=args.crm_of
                    )
                    metric_list += np.array(metric_i)
                metric_list /= len(val_loader)

                for class_i in range(num_classes - 1):
                    writer.add_scalar('results/model1_val_class_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('results/model1_val_class_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)
                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd95_1 = np.mean(metric_list, axis=0)[1]

                writer.add_scalar('results/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('results/model1_val_mean_hd95', mean_hd95_1, iter_num)
                print('Average DICE:{:.3f}'.format(performance1))
                if performance1 > best_performance1:
                    best_performance1 = performance1
                    # Seg1 is the primary network (retained for single-network inference).
                    torch.save(model1.state_dict(), os.path.join(model_dir, 'best_seg1.pth'))
                    if args.ensemble:
                        torch.save(model2.state_dict(), os.path.join(model_dir, 'best_seg2.pth'))
                    if args.crm_of:
                        torch.save(crm.state_dict(), os.path.join(model_dir, 'best_crm.pth'))
            if iter_num >= max_iterations:
                break
        if iter_num > max_iterations:
            iterator.close()
            break
        
        
    writer.close()

def tta(input, net, args, activate=True):
    """TTA with controlled augmentations using only PyTorch operations"""
    aug_predictions = []
    
    # Original
    with torch.no_grad():
        out = net(input)[0]
        if args.crm_of:
            out = args.crm_md(out)
        if not activate:
            return out
        aug_predictions.append(out)
    
    # Shifts using roll (easy to reverse)
    shifts = [(5, 0), (-5, 0), (0, 5), (0, -5)]
    for shift_y, shift_x in shifts:
        input_shifted = torch.roll(input, shifts=(shift_y, shift_x), dims=(2, 3))
        with torch.no_grad():
            out = net(input_shifted)[0]
            if args.crm_of:
                out = args.crm_md(out)
            # Reverse shift
            out = torch.roll(out, shifts=(-shift_y, -shift_x), dims=(2, 3))
            aug_predictions.append(out)
    
    # Rotations using rot90 (90 degree increments)
    for k in [1, -1]:  # 90 and -90 degrees
        input_rot = torch.rot90(input, k=k, dims=[2, 3])
        with torch.no_grad():
            out = net(input_rot)[0]
            if args.crm_of:
                out = args.crm_md(out)
            # Reverse rotation
            out = torch.rot90(out, k=-k, dims=[2, 3])
            aug_predictions.append(out)
    
    # Small scale changes using interpolate
    scales = [0.95, 1.05]
    _, _, h, w = input.shape
    for scale in scales:
        # Scale input
        new_size = (int(h * scale), int(w * scale))
        input_scaled = F.interpolate(input, size=new_size, mode='bilinear', align_corners=False)
        
        # Pad or crop to original size
        if scale < 1.0:
            # Pad to original size
            pad_h = (h - new_size[0]) // 2
            pad_w = (w - new_size[1]) // 2
            input_scaled = F.pad(input_scaled, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        else:
            # Center crop to original size
            start_h = (new_size[0] - h) // 2
            start_w = (new_size[1] - w) // 2
            input_scaled = input_scaled[:, :, start_h:start_h+h, start_w:start_w+w]
        
        with torch.no_grad():
            out = net(input_scaled)[0]
            if args.crm_of:
                out = args.crm_md(out)
            aug_predictions.append(out)
    
    # Average predictions
    avg_pred = sum(aug_predictions) / len(aug_predictions)
    
    return avg_pred

def testing(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_torch(args.seed)
    
    image_size = args.image_size
    num_classes = args.num_classes
    exp_dir = args.exp_dir
    model_dir = exp_dir + '/model'
    pred_dir = exp_dir + '/predictions'

    model_path = os.path.join(model_dir, 'best_seg1.pth')
    model = ProjectUNet('resnet50', None, classes=num_classes, deep_stem=32).cuda()
    model2 = ProjectUNet('resnet50', None, classes=num_classes, deep_stem=32).cuda()

    model.load_state_dict(torch.load(model_path, map_location=device))
    if args.ensemble:
        model2.load_state_dict(torch.load(os.path.join(model_dir, 'best_seg2.pth'), map_location=device))
        model2.eval()
    model.eval()

    args.crm_md = None
    if args.crm_of:
        crm = ClassConditionedRefinement().cuda()
        crm.load_state_dict(torch.load(os.path.join(model_dir, 'best_crm.pth'), map_location=device))
        crm.eval()
        args.crm_md = crm

    mkdir(pred_dir)

    data_list = get_image_list(args.test_data_list)

    print("using {} images for testing.".format(len(data_list['image_filenames'])))

    with torch.no_grad():
        for index in range(len(data_list['image_filenames'])):
            gt_name = data_list['label_filenames'][index]
            nib_gt = nib.load(os.path.join(args.test_data_dir, gt_name))

            gt = nib_gt.get_fdata()

            img_name = data_list['image_filenames'][index]
            nib_img = nib.load(os.path.join(args.test_data_dir, img_name))

            img = nib_img.get_fdata().astype('float32')

            clip_min = np.percentile(img, 1)
            clip_max = np.percentile(img, 99)
            img = np.clip(img, clip_min, clip_max)
            img = (img - img.min()) / float(img.max() - img.min())
            x, y, z = img.shape
            x_centre, y_centre = int(x / 2), int(y / 2)
            img = crop_image(img, x_centre, y_centre, image_size, constant_values=0)

            pred_res = torch.zeros(img.shape, dtype=torch.int8)
            for i in range(img.shape[2]):
                tmp_image = torch.from_numpy(img[:, :, i]).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
                outputs = tta(tmp_image, model, args, activate=False)
                if args.ensemble:
                    outputs2 = tta(tmp_image, model2, args, activate=False)
                    outputs = (outputs + outputs2) / 2
                tmp_prob = F.softmax(outputs, dim=1)  # 1*C*H*W
                pred_res[:, :, i] = torch.argmax(tmp_prob[0, :, :, :], dim=0)

            pred_res = crop_image(pred_res, image_size[0] // 2, image_size[1] // 2, (x, y), constant_values=0)
            pred_res = pred_res.astype('int16')
            nii_pred = nib.Nifti1Image(pred_res, None, header=nib_gt.header)
            mkdir(pred_dir)
            loc_end = img_name.find('.')
            loc_start = img_name.rfind('/')
            savedirname = os.path.join(pred_dir, img_name[:loc_start])

            mkdir(savedirname)
            pred_name = savedirname + img_name[loc_start:loc_end] + '_Pred.nii.gz'
            nib.save(nii_pred, pred_name)
    print('Finished Testing!')

if __name__ == '__main__':
    args = parse_args()
    mode_ = args.mode
    torch.cuda.set_device(args.device)
    if mode_ == 'train':
        training(args)
        testing(args)
    elif mode_ == 'test':
        testing(args)
    else:
        raise NotImplementedError