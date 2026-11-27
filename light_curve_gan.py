"""
DCGAN per generare light curve sintetiche dal dataset TESS.

La preparazione dei dati riproduce il notebook GAN_task_I_flux_errors.ipynb:
- 1000 colonne flux_0000 ... flux_0999;
- prime 936 colonne flux_err_0000 ... flux_err_0935;
- concatenazione (1000 flux + 936 errori = 1936 valori);
- reshape in immagini (N, 1, 44, 44).

Dipendenze:
    pip install numpy pandas torch pyyaml matplotlib

Esempio:
    from light_curve_gan import LightCurveGAN

    gan = LightCurveGAN("config.yaml")
    gan.load_data()
    gan.build_models()
    gan.train(epochs=15)
    synthetic_curves = gan.generate(n_samples=100)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    """Generatore DCGAN configurabile tramite il file YAML."""

    def __init__(self, nz: int, layers: list[dict]):
        super().__init__()
        modules = []
        in_channels = nz

        for layer in layers:
            out_channels = int(layer["out_channels"])
            modules.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=int(layer["kernel_size"]),
                    stride=int(layer.get("stride", 1)),
                    padding=int(layer.get("padding", 0)),
                    bias=bool(layer.get("bias", False)),
                )
            )
            if layer.get("batch_norm", False):
                modules.append(nn.BatchNorm2d(out_channels))

            activation = layer.get("activation", "relu").lower()
            if activation == "relu":
                modules.append(nn.ReLU(inplace=True))
            elif activation == "leaky_relu":
                modules.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation == "tanh":
                modules.append(nn.Tanh())
            elif activation == "sigmoid":
                modules.append(nn.Sigmoid())
            elif activation in ("none", "identity", None):
                pass
            else:
                raise ValueError(f"Attivazione Generator non supportata: {activation}")

            in_channels = out_channels

        self.main = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


class Discriminator(nn.Module):
    """Discriminatore DCGAN configurabile tramite il file YAML."""

    def __init__(self, nc: int, layers: list[dict]):
        super().__init__()
        modules = []
        in_channels = nc

        for layer in layers:
            out_channels = int(layer["out_channels"])
            modules.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=int(layer["kernel_size"]),
                    stride=int(layer.get("stride", 1)),
                    padding=int(layer.get("padding", 0)),
                    bias=bool(layer.get("bias", False)),
                )
            )
            if layer.get("batch_norm", False):
                modules.append(nn.BatchNorm2d(out_channels))

            activation = layer.get("activation", "leaky_relu").lower()
            if activation == "relu":
                modules.append(nn.ReLU(inplace=True))
            elif activation == "leaky_relu":
                modules.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation == "tanh":
                modules.append(nn.Tanh())
            elif activation == "sigmoid":
                modules.append(nn.Sigmoid())
            elif activation in ("none", "identity", None):
                pass
            else:
                raise ValueError(f"Attivazione Discriminator non supportata: {activation}")

            in_channels = out_channels

        self.main = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class LightCurveGAN:
    """
    Gestisce caricamento/preprocessing, modelli, training e generazione.

    Nota: il reshape 44x44 è quello del notebook, non una rappresentazione
    temporale naturale. In fase di generazione vengono estratti i primi 1000
    valori appiattiti come flussi e i successivi 936 come errori.
    """

    def __init__(self, config_path: str | Path):
        """Inizializza la GAN leggendo tutti i parametri dal file YAML.

        I percorsi relativi (ad esempio data.csv_path) sono risolti rispetto
        alla directory in cui si trova il file YAML.
        """
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"File YAML non trovato: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

        if not isinstance(config, dict):
            raise ValueError("Il file YAML deve contenere una mappa di configurazione.")

        required_sections = ("data", "model", "training")
        missing_sections = [key for key in required_sections if key not in config]
        if missing_sections:
            raise ValueError(
                "Sezioni mancanti nel YAML: " + ", ".join(missing_sections)
            )

        self.config = config
        data_cfg = config["data"]
        model_cfg = config["model"]
        training_cfg = config["training"]

        csv_value = data_cfg.get("csv_path")
        if not csv_value:
            raise ValueError("Specificare data.csv_path nel file YAML.")
        csv_path = Path(csv_value).expanduser()
        if not csv_path.is_absolute():
            csv_path = self.config_path.parent / csv_path
        self.csv_path = csv_path.resolve()

        # Parametri relativi alla rappresentazione dei dati.
        self.n_bins = int(data_cfg.get("n_flux", 1000))
        self.n_error_bins = int(data_cfg.get("n_errors", 936))
        self.image_size = int(data_cfg.get("image_size", 44))
        self.n_features = self.n_bins + self.n_error_bins
        if self.n_features != self.image_size ** 2:
            raise ValueError(
                f"n_flux + n_errors deve essere uguale a image_size²: "
                f"{self.n_bins} + {self.n_error_bins} != {self.image_size**2}."
            )

        # Architettura della GAN.
        self.nz = int(model_cfg["latent_dim"])
        self.nc = int(model_cfg.get("input_channels", 1))
        self.generator_layers = model_cfg["generator_layers"]
        self.discriminator_layers = model_cfg["discriminator_layers"]
        if not self.generator_layers or not self.discriminator_layers:
            raise ValueError("Le liste generator_layers e discriminator_layers non possono essere vuote.")

        # Parametri di training.
        self.batch_size = int(training_cfg.get("batch_size", 128))
        self.epochs = int(training_cfg.get("epochs", 15))
        self.lr = float(training_cfg.get("learning_rate", 0.0002))
        self.beta1 = float(training_cfg.get("beta1", 0.5))
        self.num_workers = int(training_cfg.get("num_workers", 0))
        self.ngpu = int(training_cfg.get("ngpu", 1))
        self.seed = training_cfg.get("seed", 999)

        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size ed epochs devono essere positivi.")
        if self.lr <= 0:
            raise ValueError("learning_rate deve essere positivo.")

        if self.seed is not None:
            torch.manual_seed(int(self.seed))
            np.random.seed(int(self.seed))

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.ngpu > 0 else "cpu"
        )

        self.data_loader: Optional[DataLoader] = None
        self.images: Optional[torch.Tensor] = None
        self.netG: Optional[Generator] = None
        self.netD: Optional[Discriminator] = None
        self.optimizerG = None
        self.optimizerD = None
        self.criterion = None

        # Parametri per riportare i dati dalla scala [-1, 1] alle unità originali.
        self.flux_min = None
        self.flux_scale = None
        self.error_min = None
        self.error_scale = None

        self.G_losses: list[float] = []
        self.D_losses: list[float] = []

    @staticmethod
    def _weights_init(module: nn.Module) -> None:
        name = module.__class__.__name__
        if "Conv" in name:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif "BatchNorm" in name:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def load_data(self) -> torch.Tensor:
        """Legge il CSV, normalizza, concatena e fa il reshape 44x44."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV non trovato: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        flux_cols = [f"flux_{i:04d}" for i in range(self.n_bins)]
        error_cols = [f"flux_err_{i:04d}" for i in range(self.n_error_bins)]

        missing = [c for c in flux_cols + error_cols if c not in df.columns]
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(
                f"Nel CSV mancano {len(missing)} colonne richieste. "
                f"Prime colonne mancanti: {preview}"
            )

        flux = df[flux_cols].to_numpy(dtype=np.float32)
        errors = df[error_cols].to_numpy(dtype=np.float32)

        if not np.isfinite(flux).all() or not np.isfinite(errors).all():
            raise ValueError("Il CSV contiene NaN o valori infiniti nelle colonne flux/errori.")

        # Normalizzazione dei flussi: percentili 0.1 e 99.9, come nel notebook.
        self.flux_min = float(np.percentile(flux, 0.1))
        flux_max = float(np.percentile(flux, 99.9))
        self.flux_scale = float(flux_max - self.flux_min)
        if self.flux_scale == 0:
            self.flux_scale = 1.0

        flux_norm = np.clip(
            2.0 * (flux - self.flux_min) / self.flux_scale - 1.0,
            -1.0,
            1.0,
        )

        # Normalizzazione separata degli errori, come nel notebook.
        self.error_min = float(errors.min())
        error_max = float(errors.max())
        self.error_scale = float(error_max - self.error_min)
        if self.error_scale == 0:
            self.error_scale = 1.0

        errors_norm = 2.0 * (errors - self.error_min) / self.error_scale - 1.0

        combined = np.concatenate((flux_norm, errors_norm), axis=1)
        if combined.shape[1] != self.image_size * self.image_size:
            raise RuntimeError(
                f"Attesi {self.image_size**2} valori per immagine, "
                f"trovati {combined.shape[1]}."
            )

        # Equivalente a reshape(-1, 1, 44, 44) nel notebook.
        self.images = torch.from_numpy(
            combined.reshape(-1, 1, self.image_size, self.image_size)
        ).float()

        dataset = TensorDataset(self.images)
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        print(f"Caricati {len(dataset)} esempi dal CSV.")
        print(f"Shape immagini: {tuple(self.images.shape)}")
        print(f"Device: {self.device}")
        return self.images

    def build_models(self) -> tuple[Generator, Discriminator]:
        """Crea Generator, Discriminator, loss e ottimizzatori."""
        self.netG = Generator(nz=self.nz, layers=self.generator_layers).to(self.device)
        self.netD = Discriminator(nc=self.nc, layers=self.discriminator_layers).to(self.device)

        self.netG.apply(self._weights_init)
        self.netD.apply(self._weights_init)

        self.criterion = nn.BCELoss()
        self.optimizerG = torch.optim.Adam(
            self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        self.optimizerD = torch.optim.Adam(
            self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        return self.netG, self.netD

    def train(self, epochs: Optional[int] = None, log_every: int = 50) -> tuple[list[float], list[float]]:
        """Allena la GAN con lo schema DCGAN del notebook."""
        if epochs is None:
            epochs = self.epochs
        if log_every < 1:
            raise ValueError("log_every deve essere almeno 1.")
        if self.data_loader is None:
            self.load_data()
        if self.netG is None or self.netD is None:
            self.build_models()

        assert self.data_loader is not None
        assert self.netG is not None and self.netD is not None

        self.netG.train()
        self.netD.train()
        self.G_losses.clear()
        self.D_losses.clear()

        for epoch in range(epochs):
            for batch_index, (real_batch,) in enumerate(self.data_loader):
                real = real_batch.to(self.device, dtype=torch.float32)
                batch = real.size(0)

                # 1. Aggiorna il discriminatore con esempi reali.
                self.netD.zero_grad(set_to_none=True)
                labels_real = torch.ones(batch, device=self.device)
                output_real = self.netD(real).view(-1)
                loss_D_real = self.criterion(output_real, labels_real)
                loss_D_real.backward()
                d_x = output_real.mean().item()

                # 2. Aggiorna il discriminatore con esempi generati.
                noise = torch.randn(batch, self.nz, 1, 1, device=self.device)
                fake = self.netG(noise)
                labels_fake = torch.zeros(batch, device=self.device)
                output_fake = self.netD(fake.detach()).view(-1)
                loss_D_fake = self.criterion(output_fake, labels_fake)
                loss_D_fake.backward()
                d_g_z1 = output_fake.mean().item()
                loss_D = loss_D_real + loss_D_fake
                self.optimizerD.step()

                # 3. Aggiorna il generatore: vuole che D classifichi fake come reali.
                self.netG.zero_grad(set_to_none=True)
                labels_generator = torch.ones(batch, device=self.device)
                output_for_G = self.netD(fake).view(-1)
                loss_G = self.criterion(output_for_G, labels_generator)
                loss_G.backward()
                d_g_z2 = output_for_G.mean().item()
                self.optimizerG.step()

                self.D_losses.append(float(loss_D.item()))
                self.G_losses.append(float(loss_G.item()))

                if batch_index % log_every == 0:
                    print(
                        f"[{epoch + 1}/{epochs}] "
                        f"[{batch_index}/{len(self.data_loader)}] "
                        f"Loss_D: {loss_D.item():.4f} "
                        f"Loss_G: {loss_G.item():.4f} "
                        f"D(x): {d_x:.4f} "
                        f"D(G(z)): {d_g_z1:.4f}/{d_g_z2:.4f}"
                    )

        return self.G_losses, self.D_losses

    @torch.no_grad()
    def generate(
        self,
        n_samples: int = 1,
        return_errors: bool = False,
        batch_size: int = 256,
    ):
        """
        Genera light curve sintetiche.

        Ritorna un array (n_samples, 1000) con i flussi in unità originali.
        Se return_errors=True, ritorna (flux, errors), con errors di lunghezza 936.
        """
        if self.netG is None:
            raise RuntimeError("Il generatore non esiste: esegui prima build_models().")
        if self.flux_min is None or self.flux_scale is None:
            raise RuntimeError("Prima di generare devi eseguire load_data().")
        if n_samples < 1:
            raise ValueError("n_samples deve essere almeno 1.")

        was_training = self.netG.training
        self.netG.eval()
        generated = []

        remaining = n_samples
        while remaining:
            current_batch = min(batch_size, remaining)
            noise = torch.randn(current_batch, self.nz, 1, 1, device=self.device)
            generated.append(self.netG(noise).cpu().numpy())
            remaining -= current_batch

        self.netG.train(was_training)

        images = np.concatenate(generated, axis=0)
        flat = images.reshape(n_samples, self.n_features)

        flux_norm = flat[:, :self.n_bins]
        flux = (flux_norm + 1.0) * self.flux_scale / 2.0 + self.flux_min

        if not return_errors:
            return flux

        if self.error_min is None or self.error_scale is None:
            raise RuntimeError("Parametri di normalizzazione errori non disponibili.")
        errors_norm = flat[:, self.n_bins:]
        errors = (errors_norm + 1.0) * self.error_scale / 2.0 + self.error_min
        return flux, errors

    def plot_lightcurves(self, lightcurves, n_curves: int = 5, separate: bool = True) -> None:
        """Visualizza un sottoinsieme di light curve 1D generate."""
        import matplotlib.pyplot as plt

        curves = np.asarray(lightcurves)
        if curves.ndim != 2:
            raise ValueError("lightcurves deve avere shape (N, punti_temporali).")
        if n_curves < 1:
            raise ValueError("n_curves deve essere almeno 1.")
        n_curves = min(n_curves, len(curves))
        if n_curves == 0:
            raise ValueError("Non ci sono light curve da visualizzare.")

        if separate:
            fig, axes = plt.subplots(n_curves, 1, figsize=(11, 2.8 * n_curves), squeeze=False)
            for i in range(n_curves):
                axes[i, 0].plot(curves[i])
                axes[i, 0].set_title(f"Light curve sintetica {i + 1}")
                axes[i, 0].set_xlabel("Time index")
                axes[i, 0].set_ylabel("Flux")
                axes[i, 0].grid(alpha=0.3)
        else:
            fig, ax = plt.subplots(figsize=(11, 5))
            for i in range(n_curves):
                ax.plot(curves[i], label=f"Curva {i + 1}")
            ax.set_xlabel("Time index")
            ax.set_ylabel("Flux")
            ax.set_title("Light curve sintetiche")
            ax.legend()
            ax.grid(alpha=0.3)

        fig.tight_layout()
        plt.show()

    def save(self, path: str | Path) -> None:
        """Salva pesi, configurazione e parametri di normalizzazione."""
        if self.netG is None or self.netD is None:
            raise RuntimeError("Non ci sono modelli da salvare.")
        checkpoint = {
            "netG": self.netG.state_dict(),
            "netD": self.netD.state_dict(),
            "config": {
                "latent_dim": self.nz, "input_channels": self.nc,
                "generator_layers": self.generator_layers,
                "discriminator_layers": self.discriminator_layers,
                "learning_rate": self.lr, "beta1": self.beta1,
            },
            "normalization": {
                "flux_min": self.flux_min,
                "flux_scale": self.flux_scale,
                "error_min": self.error_min,
                "error_scale": self.error_scale,
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Carica i pesi e la normalizzazione da un checkpoint salvato."""
        checkpoint = torch.load(path, map_location=self.device)
        if self.netG is None or self.netD is None:
            self.build_models()
        self.netG.load_state_dict(checkpoint["netG"])
        self.netD.load_state_dict(checkpoint["netD"])
        norm = checkpoint["normalization"]
        self.flux_min = norm["flux_min"]
        self.flux_scale = norm["flux_scale"]
        self.error_min = norm["error_min"]
        self.error_scale = norm["error_scale"]
