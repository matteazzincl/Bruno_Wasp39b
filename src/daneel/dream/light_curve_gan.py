import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
#import torchvision.datasets as dset
#import torchvision.transforms as transforms
#import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# Generator Code
# alreday built to read 44
class Generator(nn.Module):
    def __init__(self, params_dict):
        super(Generator, self).__init__()

        nz = params_dict.get("nz")
        ngf = params_dict.get("ngf")
        nc = params_dict.get("nc")
        image_size = params_dict.get("image_size")
        ngpu = params_dict.get("ngpu")

        self.main = nn.Sequential(
            # Input: nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 3, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 11 x 11
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 22 x 22
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()

            # 44 x 44
        )

    def forward(self, input):
        return self.main(input)

# Discriminator code
# already built for 44x44
class Discriminator(nn.Module):
    def __init__(self, params_dict):
        super(Discriminator, self).__init__()

        nc = params_dict.get("nc")
        ndf = params_dict.get("ndf")
        ngpu = params_dict.get("ngpu")
        image_size = params_dict.get("image_size")

        self.main = nn.Sequential(
            # Input: nc x 44 x 44
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf x 22 x 22
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2) x 11 x 11
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4) x 6 x 6
            nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*8) x 3 x 3
            nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()

            # Output: 1 x 1 x 1
        )

    def forward(self, input):
        return self.main(input)

class GAN_class:

    def __init__(self, params_dict):

        # Parametri relativi ai dati
        self.params_dict = params_dict
        self.csv_path = params_dict.get("csv_path")
        self.image_size = params_dict.get("image_size")
        self.n_curves = params_dict.get("n_curves")
        self.n_samples = params_dict.get("n_samples")

        self.n_bins = params_dict.get("flux_nbins") #1000
        self.n_error_bins = params_dict.get("fluxerr_nbins") #936
        self.n_features = self.n_bins + self.n_error_bins

        # Parametri relativi ai modelli
        self.nc = params_dict.get("nc")
        self.nz = params_dict.get("nz")
        self.ngf = params_dict.get("ngf")
        self.ndf = params_dict.get("ndf")

        # Parametri di training
        self.batch_size = params_dict.get("batch_size")
        self.num_epochs = params_dict.get("num_epochs")
        self.lr = params_dict.get("lr")
        self.beta1 = params_dict.get("beta1")
        self.ngpu = params_dict.get("ngpu")

        # Device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.ngpu > 0 else "cpu"
        )

        # Data loader e dati
        self.data_loader = None
        self.images = None

        # Modelli
        self.netG = None
        self.netD = None

        # Optimizers e loss
        self.optimizerG = None
        self.optimizerD = None
        self.criterion = None

        # Parametri di normalizzazione
        self.flux_center = 1.0
        self.flux_scale = None
        self.error_min = None
        self.error_scale = None

        # Loss durante il training
        self.G_losses = []
        self.D_losses = []
        
    def _weights_init(self, m):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def load_data(self):
        #loads the data from the tess_data.csv

        df = pd.read_csv(self.csv_path)
        
        #generates the name of the columns till having 1000 fluxes and 936 err_fluxes
        flux_cols = [f"flux_{i:04d}" for i in range(self.n_bins)]
        error_cols = [f"flux_err_{i:04d}" for i in range(self.n_error_bins)]

        flux = df[flux_cols].to_numpy(dtype=np.float32)
        errors = df[error_cols].to_numpy(dtype=np.float32)

        #if not np.isfinite(flux).all() or not np.isfinite(errors).all():
         #   raise ValueError("Il CSV contiene valori non validi.")

        # MODIFIED: robust flux normalization

        self.flux_scale = float(
            np.percentile(np.abs(flux - self.flux_center), 99.9)
        )

        if self.flux_scale == 0:
            raise ValueError("Flux normalization scale is zero.")

        flux_norm = (flux - self.flux_center) / self.flux_scale

        # NEW: fraction of values outside [-1, 1]
        clipped_fraction = np.mean(np.abs(flux_norm) > 1)

        # NEW: clipping
        flux_norm = np.clip(flux_norm, -1.0, 1.0)

        # NEW: normalization statistics
        print("\nNORMALIZED FLUX STATISTICS")
        print("Flux center:", self.flux_center)
        print("Flux scale:", self.flux_scale)
        print("Normalized flux min:", np.min(flux_norm))
        print("Normalized flux max:", np.max(flux_norm))
        print("Normalized flux mean:", np.mean(flux_norm))
        print("Normalized flux std:", np.std(flux_norm))
        print("Clipped fraction:", clipped_fraction * 100, "%")

        # flux_err nomalization
        self.error_min = float(errors.min())
        error_max = float(errors.max())

        self.error_scale = error_max - self.error_min

        if self.error_scale == 0:
            self.error_scale = 1.0

        errors_norm = 2 * (errors - self.error_min) / self.error_scale - 1

        # jointing fluxes and errors
        combined = np.concatenate((flux_norm, errors_norm), axis=1)

        # Reshape  44x44
        images = combined.reshape(-1, 1, self.image_size, self.image_size)

        self.images = torch.from_numpy(images).float()

        # Creation DataLoader
        dataset = TensorDataset(self.images)

        self.data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda")
        )

        #print(f"Caricati {len(dataset)} esempi.")
        #print(f"Shape immagini: {self.images.shape}")

        return self.images

    def build_models(self):

        # Creation netG netD from generator and discriminator
        self.netG = Generator(self.params_dict).to(self.device)
        self.netD = Discriminator(self.params_dict).to(self.device)

        # initialization of weights
        self.netG.apply(self._weights_init)
        self.netD.apply(self._weights_init)

        # loss function
        self.criterion = nn.BCELoss()

        # optimizers
        self.optimizerG = torch.optim.Adam(
            self.netG.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999)
        )

        self.optimizerD = torch.optim.Adam(
            self.netD.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999)
        )

        return self.netG, self.netD

    def train(self):
        #training of the GAN

        # data loading and building of models
        if self.data_loader is None:
            self.load_data()

        if self.netG is None or self.netD is None:
            self.build_models()

        self.netG.train()
        self.netD.train()

        self.G_losses = []
        self.D_losses = []

        for epoch in range(self.num_epochs):

            for batch_index, (real_batch,) in enumerate(self.data_loader):

                real = real_batch.to(self.device)
                batch = real.size(0)
                
                # Training Discriminator
                self.netD.zero_grad()

                # real examples
                labels_real = torch.ones(batch, device=self.device)
                output_real = self.netD(real).view(-1)

                loss_D_real = self.criterion(output_real, labels_real)
                loss_D_real.backward()

                d_x = output_real.mean().item()

                # generated example
                noise = torch.randn(batch, self.nz, 1, 1, device=self.device)
                fake = self.netG(noise)

                labels_fake = torch.zeros(batch, device=self.device)
                output_fake = self.netD(fake.detach()).view(-1)
    
                loss_D_fake = self.criterion(output_fake, labels_fake)
                loss_D_fake.backward()

                d_g_z1 = output_fake.mean().item()

                loss_D = loss_D_real + loss_D_fake
                self.optimizerD.step()

                # Training Generator
                self.netG.zero_grad()

                labels_generator = torch.ones(batch, device=self.device)
                output_generator = self.netD(fake).view(-1)

                loss_G = self.criterion(output_generator, labels_generator)
                loss_G.backward()

                d_g_z2 = output_generator.mean().item()

                self.optimizerG.step()

                # Saving loss
                self.D_losses.append(loss_D.item())
                self.G_losses.append(loss_G.item())

        return self.G_losses, self.D_losses

    def generate(self, return_errors=False, batch_size=256):
    #generating lightcurves
        n_samples = self.n_samples
    
        if self.netG is None:
            raise RuntimeError("Generator not builded.")

        if self.flux_center is None or self.flux_scale is None:
            raise RuntimeError("do load_data() first.")

        if n_samples < 1:
            raise ValueError("n_samples at least 1")

        # evaluation
        self.netG.eval()
    
        generated = []

        # light curves generation
        remaining = n_samples

        while remaining > 0:

            current_batch = min(batch_size, remaining)

            noise = torch.randn(
                current_batch, self.nz, 1, 1, device=self.device
            )

            with torch.no_grad():
                fake = self.netG(noise)

            generated.append(fake.cpu().numpy())

            remaining -= current_batch

        # restoring previous mode
        self.netG.train()

        # reshaping matrices to 1D
        images = np.concatenate(generated, axis=0)
        flat = images.reshape(n_samples, self.image_size**2)

        # flux and flux_err separation
        flux_norm = flat[:, :self.n_bins]

        # MODIFIED: setting fluxes to original scales
        flux = flux_norm * self.flux_scale + self.flux_center

        if not return_errors:
            return flux

        if self.error_min is None or self.error_scale is None:
            raise RuntimeError("Parametri di normalizzazione errori non disponibili.")
        
        errors_norm = flat[:, self.n_bins:]

        # setting flux_err to original scale
        errors = (errors_norm + 1) * self.error_scale / 2 + self.error_min

        return flux, errors

        
    def plot_lightcurves(self, lightcurves):
        #Plots a certain number of generated lightcurves

        import matplotlib.pyplot as plt

        curves = np.asarray(lightcurves)

        if curves.ndim != 2:
            raise ValueError("light curve shape should be (N, time_points).")

        if len(curves) == 0:
            raise ValueError("No light curve to visualize")

        n_curves = min(self.n_curves, len(curves))

        fig, axes = plt.subplots(
            n_curves, 1,
            figsize=(11, 2.8 * n_curves),
            squeeze=False
        )

        for i in range(n_curves):
            axes[i, 0].plot(curves[i])
            axes[i, 0].set_title(f"GAN generated lightcurve {i + 1}")
            axes[i, 0].set_xlabel("Time index")
            axes[i, 0].set_ylabel("Flux")
            axes[i, 0].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig("generated_lightcurves.png", dpi=300)
        plt.show()

    def run_GAN(self):
        #runs class till generate
        # data loading

        self.load_data()
        print("Data loaded")

        # model building
        self.build_models()
        print("Building the models")

        # Training
        self.train()
        print("Training the GAN")

        # Generating light curve
        lightcurves = self.generate()
        print("Generating lightcurves")

        return lightcurves