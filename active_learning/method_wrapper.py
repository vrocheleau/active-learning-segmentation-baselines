from torch.utils.data import DataLoader
import torch
from torch.optim import SGD, lr_scheduler, adam
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from baal.bayesian.dropout import MCDropoutModule
from utils.metrics import Evaluator
from utils.utils import ExpandedRandomSampler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class AbstractMethodWrapper:

    def __init__(self, base_model, n_classes, state_dict_path):
        self.n_classes = n_classes
        self.state_dict_path = state_dict_path
        self.base_model = base_model

    def train(self, train_ds, val_ds, epochs, batch_size, optimizer, scheduler=None, model_checkpoint=True, early_stopping=False):
        raise NotImplementedError

    def evaluate(self, loader, test=False):
        raise NotImplementedError

    def predict(self, dataset, n_predictions):
        raise NotImplementedError

    def reset_params(self):
        raise NotImplementedError


class MCDropoutUncert(AbstractMethodWrapper):

    def __init__(self, base_model, n_classes, state_dict_path):
        super().__init__(base_model, n_classes, state_dict_path)
        self.model = MCDropoutModule(base_model)
        # self.base_state_dict = deepcopy(self.model.module.state_dict())
        torch.save(self.model.state_dict(), self.state_dict_path)

    def train(self, train_ds, val_ds, epochs, batch_size, opt_sch_callable, test_ds=None, model_checkpoint=True, early_stopping=False):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)

        optimizer, scheduler = opt_sch_callable(self.model)

        train_loader = DataLoader(train_ds,
                                  batch_size=batch_size,
                                  sampler=ExpandedRandomSampler(len(train_ds), multiplier=8))
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        test_loader = None
        if test_ds is not None:
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        best_val_loss = float('inf')
        best_state_dict = self.model.state_dict()

        writer = SummaryWriter()

        for epoch in range(epochs):
            self.model.train()

            print('Epoch {} / {}'.format(epoch + 1, epochs))

            for images, masks, _, _ in tqdm(train_loader, ncols=100, desc='Training'):
                images, masks = images.to(device), masks.squeeze(1).to(device, non_blocking=True)
                class_masks = (masks != 0).long()

                out = self.model(images)

                # loss = self.criterion(out, class_masks)
                loss = F.cross_entropy(out, class_masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()
            val_metrics = self.evaluate(val_loader, test=False)
            if val_metrics['losses'].mean() <= best_val_loss and model_checkpoint:
                best_val_loss = val_metrics['losses'].mean()
                best_state_dict = deepcopy(self.model.state_dict())

            if test_loader is not None:
                test_metrics = self.evaluate(test_loader, test=True)
                print('Test loss: {}'.format(test_metrics['losses'].mean()))
                print('Test mean dice: {}'.format(test_metrics['mean_dice'].mean()))
                writer.add_scalar('Loss/test', test_metrics['losses'].mean(), epoch)
                writer.add_scalar('Mean dice/test', test_metrics['mean_dice'], epoch)

            print('Val loss: {}'.format(val_metrics['losses'].mean()))
            print('Val mean dice: {}'.format(val_metrics['mean_dice'].mean()))
            writer.add_scalar('Loss/val', val_metrics['losses'].mean(), epoch)
            writer.add_scalar('Mean dice', val_metrics['mean_dice'], epoch)

        self.model.load_state_dict(best_state_dict)

    def evaluate(self, loader, test=False):
        self.model.eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        all_labels = []
        all_losses = []
        all_seg_preds = []
        all_dices = []
        all_ious = []
        evaluator = Evaluator(2)
        image_evaluator = Evaluator(2)

        pbar = tqdm(loader, ncols=80, desc='Test' if test else 'Validation')
        with torch.no_grad():
            for image, mask, label, f_name in pbar:
                image, mask = image.to(device), mask.squeeze(1).to(device, non_blocking=True)
                class_masks = (mask != 0).long()

                seg_logits = self.model(image)
                loss = F.cross_entropy(seg_logits, class_masks).item()

                seg_preds = seg_logits.argmax(1)
                evaluator.add_batch(class_masks, seg_preds)
                image_evaluator.add_batch(class_masks, seg_preds)
                dices = image_evaluator.dice()
                ious = image_evaluator.intersection_over_union()
                image_evaluator.reset()

                all_labels.append(label[0])
                all_losses.append(loss)
                all_dices.append(dices.cpu())
                all_ious.append(ious.cpu())
                all_seg_preds.append(seg_preds.squeeze(0).byte().cpu().numpy().astype('bool'))

            all_labels = np.array(all_labels)
            all_losses = np.array(all_losses)
            all_dices = torch.stack(all_dices, 0)
            all_ious = torch.stack(all_ious, 0)

        dices = evaluator.dice()
        ious = evaluator.intersection_over_union()

        metrics = {
            'images_path': loader.dataset.rows,
            'labels': all_labels,
            'losses': all_losses,
            'dice_background_per_image': all_dices[:, 0].numpy(),
            'mean_dice_background': all_dices[:, 0].numpy().mean(),
            'dice_background': dices[0].item(),
            'dice_per_image': all_dices[:, 1].numpy(),
            'mean_dice': all_dices[:, 1].numpy().mean(),
            'dice': dices[1].item(),
            'iou_background_per_image': all_ious[:, 0].numpy(),
            'mean_iou_background': all_ious[:, 0].numpy().mean(),
            'iou_background': ious[0].item(),
            'iou_per_image': all_ious[:, 1].numpy(),
            'mean_iou': all_ious[:, 1].numpy().mean(),
            'iou': ious[1].item(),
        }

        return metrics

    def predict(self, dataset, n_predictions):
        self.model.eval()
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        predictions = []
        with torch.no_grad():
            for image, mask, lbl, name in tqdm(loader, ncols=80, desc='MC predictions'):
                image = image.to(device)

                preds = [self.model(image).squeeze(0) for _ in range(n_predictions)]
                stack = torch.stack(preds, dim=-1).cpu().numpy()

                # Return prediction as tuples of (mc_predictions, input_image, gt_mask, file_name, label)
                stack_tuple = (stack, image.cpu().numpy(), mask.numpy(), name, lbl.item())
                predictions.append(stack_tuple)

        return predictions

    def reset_params(self):
        self.model = MCDropoutModule(self.base_model)
        self.model.load_state_dict(torch.load(self.state_dict_path))
        # self.base_state_dict = deepcopy(self.model.module.state_dict())