import torch
from tqdm import tqdm, trange

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        elem = {k: v[index] for k, v in self.dataset.items()}
        if self.transform is not None:
            elem = {k: self.transform(v) for k, v in elem.items()}
        return elem

    def __len__(self):
        return len(self.dataset[list(self.dataset.keys())[0]])

class Trainer:
    def __init__(self, model, data, lr, wd, scheduler=None, device=None):
        self.model = model
        self.data = data
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = scheduler
        self.device = device if device else torch.device('cpu')

    def train(self, num_epochs):
        losses = []
        self.model.train()
        for epoch in tqdm(range(num_epochs)):
            for batch in self.data:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                logits, loss = self.model(**batch)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                losses.append(loss.item())
        return torch.tensor(losses)

class LMTrainer(Trainer):
    class OffsetDataLoader:
        def __init__(self, data, pad_token):
            self.data = data
            self.pad_token = pad_token

        def __iter__(self):
            for batch in self.data:
                batch = {k: v.clone() for k, v in batch.items()}
                target_ids = batch['input_ids'][:, 1:]
                target_ids_postfix = torch.full((target_ids.shape[0],), self.pad_token, dtype=target_ids.dtype)
                batch['target_ids'] = torch.cat([target_ids, target_ids_postfix.unsqueeze(1)], dim=1)
                batch['target_mask'] = torch.ones_like(batch['target_ids'], dtype=torch.long)
                batch['target_mask'][:, -1] = 0
                yield batch

    def __init__(self, model, data, lr, wd, pad_token, scheduler=None, device=None):
        data = self.OffsetDataLoader(data, pad_token)
        super().__init__(model, data, lr, wd, scheduler, device)

class MLMTrainer(Trainer):
    class MaskingDataLoader:
        def __init__(self, data, mask_ratio, mask_token):
            self.data = data
            self.mask_ratio = mask_ratio
            self.mask_token = mask_token

        def __iter__(self):
            for batch in self.data:
                batch = {k: v.clone() for k, v in batch.items()}
                mask = (torch.rand(batch['input_ids'].shape) < self.mask_ratio)
                batch['target_ids'] = batch['input_ids'].clone()
                batch['input_ids'][mask] = self.mask_token
                batch['target_mask'] = mask.long()
                yield batch

    def __init__(self, model, data, lr, wd, mask_ratio, mask_token, scheduler=None, device=None):
        self.mask_ratio = mask_ratio
        data = self.MaskingDataLoader(data, mask_ratio, mask_token)
        super().__init__(model, data, lr, wd, scheduler, device)

class MLMEvaluator:
    def __init__(self, model, sampler, tokenizer):
        self.model = model
        self.sampler = sampler
        self.tokenizer = tokenizer

    def evaluate(self, masked_data, labels):
        self.model.eval()
        accuracy = 0.
        for i in trange(len(masked_data['input_ids'])):
            data = {'input_ids': masked_data['input_ids'][i]}
            label = {'input_ids': labels['input_ids'][i]}
            mask = (data['input_ids'] == self.tokenizer.mask_token_id).long()
            n_samples = mask.sum().item()
            pred = self.sampler.sample(data, n_samples)
            pred = pred[mask.bool()]
            label = label['input_ids'][mask.bool()]
            accuracy += torch.all(pred == label)
        return accuracy / len(masked_data['input_ids'])

class LMEvaluator:
    def __init__(self, model, sampler, tokenizer):
        self.model = model
        self.sampler = sampler
        self.tokenizer = tokenizer

    def evaluate(self, padded_data, labels):
        self.model.eval()
        accuracy = 0.
        for i in trange(len(padded_data)):
            data = padded_data[i]
            label = labels[i]
            mask = ((data['input_ids'] == self.tokenizer.pad_token_id) & (label['input_ids'] != self.tokenizer.pad_token_id)).long()
            n_samples = mask.sum().item()
            pred = self.sampler.sample(data, n_samples)
            pred = pred[mask.bool()]
            label = label['input_ids'][mask.bool()]
            accuracy += torch.all(pred == label)
        return accuracy / len(padded_data)


class MLMLocationAndTokenArgmaxSampler:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device else torch.device('cpu')

    def sample(self, tokens, num_samples=1):
        if len(tokens['input_ids'].shape) != 1:
            raise ValueError("must pass a single sequence of tokens to sample from")
        with torch.no_grad():
            input_ids = tokens['input_ids'].clone().to(self.device)
            for i in range(num_samples):
                logits = self.model(input_ids=input_ids.unsqueeze(0))[0][0]
                mask = -torch.inf * (1-(input_ids == self.tokenizer.mask_token_id).float())
                logits = logits + mask.unsqueeze(-1)
                best_logit_index = logits.argmax()
                best_token = best_logit_index % logits.shape[-1]
                best_position = best_logit_index // logits.shape[-1]
                input_ids[best_position] = best_token
        return input_ids.cpu()

class LMNextTokenArgmaxSampler:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device else torch.device('cpu')

    def sample(self, tokens, num_samples=1):
        if len(tokens['input_ids'].shape) != 1:
            raise ValueError("must pass a single sequence of tokens to sample from")
        with torch.no_grad():
            input_ids = tokens['input_ids'].clone().to(self.device)
            matches = (input_ids == self.tokenizer.pad_token_id).nonzero()[:, 0]
            if len(matches) < num_samples:
                input_ids = torch.cat([input_ids, torch.tensor([self.tokenizer.pad_token_id]*num_samples, device=self.device)])
                matches = (input_ids == self.tokenizer.pad_token_id).nonzero()
            for i in range(num_samples):
                logits = self.model(input_ids=input_ids.unsqueeze(0))[0][0]
                best_token = logits[matches[i]-1].argmax()
                input_ids[matches[i]] = best_token
        return input_ids.cpu()
