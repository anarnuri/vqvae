from models import VectorQuantizedVAE
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from dataset import CustomDataset
import torchvision.transforms as transforms
from constants import *

torch.set_float32_matmul_precision('medium')
    
bs = batch_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = CustomDataset(img_path='./INPUT_YOUR_IMG_PATH', transform=transforms.Compose([transforms.ToTensor(), 
                                                                                        transforms.Grayscale(),
                                                                                        ]))

dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=bs)
val_loader = DataLoader(val_dataset, batch_size=bs)

model = VectorQuantizedVAE(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay)

checkpoint_callback = ModelCheckpoint(
    dirpath='weights/',
    monitor='val_recon_loss',
    filename='{epoch}', 
    save_top_k=1)  

wandb_logger = WandbLogger(project='vqvae')

if torch.cuda.device_count() == 1: 
    trainer = pl.Trainer(logger=wandb_logger, accelerator="gpu", max_epochs=-1, callbacks=[checkpoint_callback])
elif torch.cuda.device_count() > 1:
    trainer = pl.Trainer(logger=wandb_logger, accelerator="gpu", devices=-1, max_epochs=-1, strategy="ddp", callbacks=[checkpoint_callback])
else:
    trainer = pl.Trainer(logger=wandb_logger, accelerator="cpu", callbacks=[checkpoint_callback])
    
# Train the model
trainer.fit(model, train_loader, val_loader)
