from torch.utils.data import DataLoader
import torch
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from baal.bayesian.dropout import MCDropoutModule
from utils.metrics import Evaluator

class AbstractMethodWrapper:

    def __init__(self, base_model, n_classes, criterion, **kwargs):
        self.base_model = base_model
        self.n_classes = n_classes
        self.criterion = criterion
        self.kwargs = kwargs

    def train(self, train_ds, val_ds, epochs, batch_size, model_checkpoint=True, early_stopping=False):
        raise NotImplementedError

    def evaluate(self, loader, test=False):
        raise NotImplementedError

    def predict(self, dataset, n_predictions):
        raise NotImplementedError

    def reset_params(self):
        raise NotImplementedError


class MCDropout_Uncert(AbstractMethodWrapper):

    def __init__(self, base_model, n_classes, **kwargs):
        super().__init__(base_model, n_classes, **kwargs)
        self.base_model = base_model
        self.mc_model = MCDropoutModule(base_model)
        self.base_state_dict = deepcopy(self.mc_model.state_dict())

    def train(self, train_ds, val_ds, epochs, batch_size, model_checkpoint=True, early_stopping=False):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = torch.nn.DataParallel(self.mc_model)
        model.to(device)
        model.train()

        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        best_state_dict = self.base_state_dict
        # for epoch in tqdm(range(epochs), ncols=100, desc='Training'):
        for epoch in range(epochs):

            print('Epoch {} / {}'.format(epoch, epochs))

            # for images, masks, _, _ in train_loader:
            for images, masks, _, _ in tqdm(train_loader, ncols=100, desc='Training'):
                images, masks = images.to(device), masks.squeeze(1).to(device, non_blocking=True)
                class_masks = (masks != 0).long()

                out = model(images)

                loss = self.criterion(out, class_masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            val_metrics = self.evaluate(val_loader, test=False)
            if val_metrics['losses'].mean() <= best_val_loss and model_checkpoint:
                best_val_loss = val_metrics['losses'].mean()
                best_state_dict = deepcopy(model.state_dict())
                if early_stopping:
                    break
            print('Val loss: {}'.format(val_metrics['losses'].mean()))
            print('Val mean dice: {}'.format(val_metrics['mean_dice'].mean()))
        model.load_state_dict(best_state_dict)
        self.mc_model.load_state_dict(best_state_dict)
        self.base_model.load_state_dict(best_state_dict)


    def evaluate(self, loader, test=False):
        self.base_model.eval()
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

                seg_logits = self.base_model(image)
                # loss = nn.CrossEntropyLoss(seg_logits, t_segmentation)
                loss = self.criterion(seg_logits, class_masks).item()

                seg_probs = torch.softmax(seg_logits, 1)
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