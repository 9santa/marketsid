import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        num_codebooks: int = 3,
        codebook_size: int = 256,
        beta: float = 0.25,
    ):
        super().__init__()

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.beta = beta

        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, dim) for _ in range(num_codebooks)]
        )

        for codebook in self.codebooks:
            nn.init.uniform_(
                codebook.weight,
                -1.0 / codebook_size,
                1.0 / codebook_size,
            )

    def forward(self, z):
        """
        z: [B, D]
        """
        residual = z
        quantized = torch.zeros_like(z)
        codes = []

        codebook_loss = z.new_zeros(())
        commitment_loss = z.new_zeros(())

        for codebook in self.codebooks:
            weights = codebook.weight  # [codebook_size, D]

            # Squared Euclidean distances
            distances = (
                residual.square().sum(dim=-1, keepdim=True)  # (B,1)
                - 2 * residual @ weights.T  # (B, K)
                + weights.square().sum(dim=-1)  # (K,)
            )  # [B, codebook_size]

            indices = distances.argmin(dim=-1)  # (B,)
            # Select closest codebook vectors for each batch item
            q = codebook(indices)

            codes.append(indices)

            codebook_loss = codebook_loss + F.mse_loss(q, residual.detach())

            commitment_loss = commitment_loss + F.mse_loss(residual, q.detach())

            # Accumulate attached q for STE
            quantized = quantized + q

            # Detach q when updating the residual to prevent gradient leekage
            residual = residual - q.detach()

        # Straight-Trough Estimator
        z_q = z + (quantized - z).detach()

        loss = codebook_loss + self.beta * commitment_loss

        codes = torch.stack(codes, dim=-1)

        return z_q, codes, loss


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_codebooks: int = 3,
        codebook_size: int = 256,
        beta: float = 0.25,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.quantizer = ResidualQuantizer(
            dim=latent_dim,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            beta=beta,
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)

        z_q, codes, vq_loss = self.quantizer(z)

        x_hat = self.decoder(z_q)

        reconstruction_loss = F.mse_loss(x_hat, x)

        loss = reconstruction_loss + vq_loss

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "vq_loss": vq_loss,
            "codes": codes,
            "reconstruction": x_hat,
        }

    @torch.no_grad()
    def encode_codes(self, x):
        z = self.encoder(x)
        _, codes, _ = self.quantizer(z)
        return codes
