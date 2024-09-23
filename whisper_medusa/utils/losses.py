import torch
import torch.nn as nn


class MedusaCrossEntropyLoss(nn.Module):
    def __init__(self, loss_on_original=False):
        super(MedusaCrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.loss_on_original = loss_on_original

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass for calculating the cross-entropy loss across multiple Medusa heads.

        Args:
            logits (torch.Tensor): A tensor of shape (num_heads, batch_size, seq_length, vocab_size)
                representing the predicted logits from the model for each Medusa head.
            labels (torch.Tensor): A tensor of shape (batch_size, seq_length) containing the true
                labels for each input in the batch.

        Returns:
            torch.Tensor: A tensor containing the stacked cross-entropy losses for each head,
            with one loss per Medusa head.

        Process:
            - If `self.loss_on_original` is True, the loss is calculated for the base head (original).
            - If False, it starts from the Medusa heads without including the base head.
            - For each Medusa head, the logits and labels are adjusted (shifted) to align the predictions
            with the correct labels, given that each head predicts a different portion of the sequence.
            - Loss is computed using cross-entropy (`self.ce`) for each head, and losses are appended to
            a list, which is returned as a stacked tensor.

        Special Cases:
            - If the sequence length is smaller than the number of heads, the loop breaks to avoid NaN values.
            - The logits and labels are reshaped to ensure compatibility with the cross-entropy calculation.
        """
        loss = []
        num_heads = logits.shape[0]
        if self.loss_on_original:
            orig_logits = logits[0, :, :].contiguous()
            orig_labels = labels.contiguous()
            orig_logits = orig_logits.view(-1, logits.shape[-1])
            orig_labels = orig_labels.view(-1)
            orig_labels = orig_labels.to(orig_logits.device)
            loss_0 = self.ce(orig_logits, orig_labels)
            loss.append(loss_0)
            start_pos = 1
            shift_idx = 0
        else:
            start_pos = 0
            shift_idx = 1

        for i in range(start_pos, num_heads):
            medusa_logits = logits[i, :, : -(shift_idx + i)].contiguous()
            medusa_labels = labels[..., shift_idx + i :].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)
            loss_i = self.ce(medusa_logits, medusa_labels)
            if torch.isnan(
                loss_i
            ).any():  # the total length of the sequence is less than the number of heads
                break
            loss.append(loss_i)
        return torch.stack(loss)


class MedusaKLDivLoss(nn.Module):
    def __init__(self, lamda=0.01, loss_on_original=False):
        super(MedusaKLDivLoss, self).__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.loss_on_original = loss_on_original
        self.lamda = lamda

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            logits: (num_heads, batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len, vocab_size)
        """
        # todo: add reduction option
        loss = []
        labels = torch.softmax(labels, dim=-1)
        logits = torch.log_softmax(logits, dim=-1)
        num_heads = logits.shape[0]
        if self.loss_on_original:
            orig_logits = logits[0, :, :, :].contiguous()
            orig_labels = labels.contiguous()

            orig_labels = orig_labels.to(orig_logits.device)
            loss_0 = self.kl(orig_logits, orig_labels) * self.lamda
            loss.append(loss_0)
            start_pos = 1
            shift_idx = 0
        else:
            start_pos = 0
            shift_idx = 1

        for i in range(start_pos, num_heads):
            medusa_logits = logits[i, :, : -(shift_idx + i)].contiguous()
            medusa_labels = labels[:, shift_idx + i :, :].contiguous()

            medusa_labels = medusa_labels.to(medusa_logits.device)
            if medusa_logits.shape[1] == 0:
                break
            loss_i = self.kl(medusa_logits, medusa_labels) * self.lamda
            loss.append(loss_i)
        return torch.stack(loss)
