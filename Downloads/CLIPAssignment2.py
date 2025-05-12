import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer
from torchvision.models import resnet50
import os


class SimpleCLIP(nn.Module):
    def __init__(self):
        super(SimpleCLIP, self).__init__()
        self.image_encoder = resnet50(pretrained=True)
        self.text_encoder = GPT2Model.from_pretrained('gpt2')
        self.image_projection = nn.Linear(2048, 512)
        self.text_projection = nn.Linear(768, 512)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, image, text):
        image_features = self.image_encoder(image)
        image_features = image_features.view(image_features.size(0), -1)
        text_outputs = self.text_encoder(input_ids=text['input_ids'], attention_mask=text['attention_mask'])

        # Use the hidden state of the last token for text features
        text_features = text_outputs.last_hidden_state[:, -1, :]

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Normalize the embeddings
        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)

        # Calculate cosine similarity and apply temperature scaling
        logits = torch.matmul(image_embeddings, text_embeddings.transpose(0, 1))
        logits = logits * self.temperature

        return logits


class Flickr30KDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, captions_file, transform=None):
        self.img_dir = img_dir
        self.captions_file = captions_file
        self.transform = transform
        self.images = []
        self.captions = []

        with open(captions_file, 'r') as f:
            for line in f:
                img_name, comment_number, caption = line.strip().split(',', 2)
                self.images.append(img_name)
                self.captions.append(caption)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def text_prompt(self, caption):
        prompt = f"An image showing:{caption}. This is"
        return prompt

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        caption = self.captions[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        text_prompt_value = self.text_prompt(caption)
        encoded = self.tokenizer(text_prompt_value, padding='max_length', truncation=True, max_length=50,
                                 return_tensors='pt')

        # Ensure tensor is moved to correct device and remove batch dimension
        text_inputs = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }

        return image, text_inputs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleCLIP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = Flickr30KDataset(img_dir='/root/flickr30k_new/flickr30k_images',
                                 captions_file='/root/flickr30k_new/captions.txt',
                                 transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Ensure model, data, and labels are moved to the correct device
def train_with_single_gpu():
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, texts in train_dataloader:
            images = images.to(device)
            input_ids = texts['input_ids'].to(device)
            attention_mask = texts['attention_mask'].to(device)

            optimizer.zero_grad()

            logits = model(images, {'input_ids': input_ids, 'attention_mask': attention_mask})
            labels = torch.arange(logits.shape[0], device=device)

            # Ensure labels are on the correct device and have the correct data type
            labels = labels.to(device).long()

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += logits.size(0)

        accuracy = correct / total
        print(f'Accuracy: {accuracy * 100:.2f}%')

        print(f"Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_dataloader)}")

    print("Training complete.")


if __name__ == '__main__':
    train_with_single_gpu()
