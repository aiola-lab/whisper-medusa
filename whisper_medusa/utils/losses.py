import torch
import torch.nn as nn


class MedusaCrossEntropyLoss(nn.Module):
    def __init__(self, loss_on_original=False):
        super(MedusaCrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.loss_on_original = loss_on_original

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            logits: (num_heads, batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len)
        """
        # todo: add reduction option
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
