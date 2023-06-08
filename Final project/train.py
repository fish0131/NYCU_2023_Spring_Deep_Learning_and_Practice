import os
import csv
import random
import numpy as np
import argparse
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision.utils import make_grid, save_image
from torchmetrics.image.fid import FrechetInceptionDistance


from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler
from model import MyDDPMPipeline, MyConditionedUNet, MyUNet2DModel

from utils import plot_result

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default="logs/lsun", type=str)
parser.add_argument('--figure_dir', default="figures", type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--eval_batch_size', default=200, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--num_epochs', default=100, type=int) 
parser.add_argument('--sample_size', default=64, type=int)
parser.add_argument('--num_workers', default=4, type=int) 
parser.add_argument('--beta_schedule', default="squaredcos_cap_v2", type=str)
parser.add_argument('--predict_type', default="epsilon", type=str)
parser.add_argument('--block_dim', default=128, type=int)
parser.add_argument('--layers_per_block', default=2, type=int)
parser.add_argument('--lr_warmup_steps', default=500, type=int)
parser.add_argument('--mixed_precision', default="fp16", type=str)
parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)
args = parser.parse_args()


# set reproducibility
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("> Using device: {}".format(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

name = 'lr={:.5f}-lr_warmup={}-block_dim={}-layers={}-schedule={}-predict_type={}'.format(args.lr, args.lr_warmup_steps, args.block_dim, args.layers_per_block, args.beta_schedule, args.predict_type)
args.log_dir = './%s/%s' % (args.log_dir, name)
args.figure_dir = '%s/%s' % (args.log_dir, args.figure_dir)

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.figure_dir, exist_ok=True)

with open('{}/train_record.txt'.format(args.log_dir), 'w') as train_record:
    train_record.write('args: {}\n'.format(args))

sample_size = args.sample_size
block_dim = args.block_dim
layers = args.layers_per_block
num_epochs = args.num_epochs
num_workers = args.num_workers
lr = args.lr
batch_size = args.batch_size
eval_batch_size = args.eval_batch_size

model = MyConditionedUNet(
    sample_size=sample_size,       # the target image resolution
    in_channels=3,                      # additional input channels for class condition
    out_channels=3,
    layers_per_block=layers,
    block_out_channels=(block_dim, block_dim, block_dim*2, block_dim*2, block_dim*4, block_dim*4),
    down_block_types=(
        "DownBlock2D",          # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",      # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",        # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",            # a regular ResNet upsampling block
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    num_class_embeds=4,
)

# prediction_type: 
# 'epsilon'(default): predicting the noise of the diffusion process
# 'sample': directly predicting the noisy sample
# beta_schedule:32
# 'linear'(default), squaredcos_cap_v2(cosine)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type=args.predict_type, beta_schedule=args.beta_schedule) 

transform = transforms.Compose([
    transforms.Resize((sample_size, sample_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

def to_onehot(label, class_num=4):
    onehot_label = np.zeros(class_num, dtype=np.float32)
    onehot_label[label] = 1.
    return onehot_label

# train_data = datasets.CIFAR10(root='dataset/cifar10', transform=transform, target_transform=to_onehot, train=True, download=True)
# train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)

classes = ['church_outdoor_train', 'classroom_train', 'conference_room_train'] 

train_data = datasets.LSUN(root='./dataset/lsun', classes=classes, transform=transform, target_transform=to_onehot)
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)

eval_sampler = RandomSampler(train_data, replacement=True, num_samples=10000)
eval_loader = DataLoader(train_data, batch_size=eval_batch_size, sampler=eval_sampler, num_workers=num_workers, pin_memory=True)

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps,
    num_training_steps=len(train_loader) * num_epochs,
)

state = {
    "model": model.state_dict(),
}

# Accelerator
accelerator = Accelerator(
    mixed_precision=args.mixed_precision,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    log_with="tensorboard",
    project_dir=os.path.join(args.log_dir, "logging"),
)

if accelerator.is_main_process:
    accelerator.init_trackers("train_example")

model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_loader, lr_scheduler
)

def calculate_fid(loader, fake):
    
    # real_images = real.to(device)
    fake_images = fake.to(device)
    fid = FrechetInceptionDistance(normalize=True).to(device)
    for x, y in eval_loader:
        real_images = x.to(device)
        fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    return float(fid.compute())


@torch.no_grad()
def evaluate(args, epoch, pipeline: MyDDPMPipeline, cfg_scale = None):

    def unnormalize_to_zero_to_one(t):
        return torch.clamp((t + 1) * 0.5, min=0., max=1.)

    # class_labels = np.zeros((100, 10))
    # class_labels[np.arange(100), np.repeat(np.arange(10), 10)] = 1. # 10000 images, 3 for each classes
    class_labels = np.zeros((200, 4))
    class_labels[np.arange(200), np.repeat(np.arange(4), 50)] = 1. # 10000 images, 3 for each classes

    cond = torch.from_numpy(class_labels).to(device)
    images = pipeline(
        batch_size=eval_batch_size,
        condition=cond,
        generator=torch.cuda.manual_seed(args.seed),
        cfg_scale=cfg_scale

    ).images

    # Make a grid out of the images
    images = unnormalize_to_zero_to_one(images)

    image_grid = make_grid(images, nrow=25)

    # Save the imagespipeline
    cfg_prefix = '' if cfg_scale is None else f'cfg{cfg_scale}_'
    save_image(image_grid, f"{args.figure_dir}/{cfg_prefix}{epoch:06d}.png")

    # for i, (x, class_label) in enumerate(eval_loader):
    #     real_images = x
    
    fid_score = calculate_fid(eval_loader, images)

    return fid_score


def train():

    losses = []
    global_step = 0

    total_iter = (len(train_loader)/batch_size)*num_epochs

    # start training
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        total_loss = 0
        for i, (x, class_label) in enumerate(train_loader):

            x, class_label = x.to(device), class_label.to(device)

            cond_labels = None if random.random() < 0.0 else class_label

            # sample some noise
            noise = torch.randn_like(x)

            # sample random timesteps
            timesteps = torch.randint(0, 1000, (x.shape[0],)).long().to(device)

            # add noise to the image and get the noisy image
            # (this is the forward diffusion process)
            noisy_image = noise_scheduler.add_noise(x, noise, timesteps)

            with accelerator.accumulate(model):
                # get the model prediction
                noise_pred = model(noisy_image, timesteps, cond_labels).sample

                # calculate the loss
                loss = loss_fn(noise_pred, noise)
                total_loss += loss.item()
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {"loss": total_loss / (i+1), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                pipeline = MyDDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                if global_step % 5000 == 0 or global_step == total_iter - 1:
                    fid_score = evaluate(args, epoch, pipeline)
                    pipeline.save_pretrained(args.log_dir)
                    torch.save(state, os.path.join(args.log_dir, "model_{}.pth".format(global_step)))
                    print(f"Save epoch#{epoch} model to {str(args.log_dir)}")

        losses.append(total_loss / len(train_loader))

        print("Epoch {}/{}, loss: {}".format(epoch+1, num_epochs, losses[-1]))

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        # if accelerator.is_main_process:
        #     pipeline = MyDDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            # if epoch % 2 == 0 or epoch == num_epochs - 1:
            # fid_score = evaluate(args, epoch, pipeline)
        
        with open('{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('[Epoch: %02d] loss: %.5f | FID: %.5f | lr: %f\n' % (epoch, losses[-1], fid_score, lr_scheduler.get_last_lr()[0])))

    plot_result(losses, args)



if __name__ == "__main__":
    train().to(device)
