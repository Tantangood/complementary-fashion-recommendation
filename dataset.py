import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from transformers import BertTokenizer
import torchvision.transforms as transforms

class DatasetLoader(Dataset):
    def __init__(self, args, split, task, transform):
        super(DatasetLoader, self).__init__()
        self.args = args
        self.split = split
        self.task = task
        self.transform = transform
        self.image_path = os.path.join(args.datadir, 'polyvore_outfits', 'images')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.per_outfit = 16 if args.polyvore_split == 'disjoint' else 19
        rootdir = os.path.join(args.datadir, 'polyvore_outfits', args.polyvore_split)
        data_json = os.path.join(
            rootdir,
            f"{'fill_in_blank' if task == 'fitb' else 'compatibility'}_{split}_new.json"
        )
        self.data = json.load(open(data_json, 'r'))

    def load_img(self, item_id):
        img_path = os.path.join(self.image_path, f'{item_id}.jpg')
        try:
            img = Image.open(img_path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error loading image {item_id}: {e}")
            return torch.zeros(3, 96, 96)

    def tokenize_text(self, text):
        if not isinstance(text, str) or text.strip() == "":
            text = "placeholder"
        encodings = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=32,
            return_tensors='pt'
        )
        return encodings['input_ids'][0], encodings['attention_mask'][0]

    def fitb(self, sample):
        imgs_list = [torch.zeros(self.per_outfit, 3, 96, 96) for _ in range(4)]
        input_ids_list = [torch.zeros(self.per_outfit, 32, dtype=torch.long) for _ in range(4)]
        attention_mask_list = [torch.zeros(self.per_outfit, 32, dtype=torch.long) for _ in range(4)]

        length = len(sample['question'])
        for i in range(length):
            im = sample['question'][i]['im']
            text = sample['question'][i]['text']
            img = self.load_img(im)
            input_ids, attention_mask = self.tokenize_text(text)
            for j in range(4):
                imgs_list[j][i] = img
                input_ids_list[j][i] = input_ids
                attention_mask_list[j][i] = attention_mask

        length += 1
        for j, ans in enumerate(sample['answers']):
            im = ans['im']
            text = ans['text']
            imgs_list[j][length - 1] = self.load_img(im)
            input_ids, attention_mask = self.tokenize_text(text)
            input_ids_list[j][length - 1] = input_ids
            attention_mask_list[j][length - 1] = attention_mask

        answer = sample['label']
        return imgs_list, input_ids_list, attention_mask_list, length, answer

    def compatibility(self, sample):
        imgs = torch.zeros(self.per_outfit, 3, 96, 96)
        input_ids = torch.zeros(self.per_outfit, 32, dtype=torch.long)
        attention_mask = torch.zeros(self.per_outfit, 32, dtype=torch.long)

        length = len(sample['items'])
        for i in range(length):
            im = sample['items'][i]['im']
            text = sample['items'][i]['text']
            imgs[i] = self.load_img(im)
            input_ids[i], attention_mask[i] = self.tokenize_text(text)

        label = int(sample['label'])
        return imgs, input_ids, attention_mask, length, label

    def __getitem__(self, index):
        sample = self.data[index]
        if self.task == 'fitb':
            imgs_list, input_ids_list, attention_mask_list, length, answer = self.fitb(sample)
            return imgs_list, input_ids_list, attention_mask_list, length, answer
        else:
            imgs, input_ids, attention_mask, length, label = self.compatibility(sample)
            return imgs, input_ids, attention_mask, length, label

    def __len__(self):
        return len(self.data)
