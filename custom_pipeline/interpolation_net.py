
import torch
import torch.nn as nn


# ============================================================
# Basic convolution block
# ============================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels
            ),

            nn.SiLU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels
            ),

            nn.SiLU()
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# InterpolationNet
# ============================================================

class InterpolationNet(nn.Module):
    """
    Small latent-space interpolation network.

    Inputs
    ------

    latent_a:
        [B, 4, 32, 32]

    latent_b:
        [B, 4, 32, 32]

    fused_latent:
        [B, 4, 32, 32]

    background_mask:
        [B, 32, 32]

    foreground_mask:
        [B, 32, 32]

    residual_mask:
        [B, 32, 32]

    Output
    ------

    predicted_latents:
        [B, 4, 16, 32, 32]
    """

    def __init__(
        self,
        latent_channels=4,
        hidden_channels=64,
        num_frames=16
    ):

        super().__init__()

        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.num_frames = num_frames

        # ----------------------------------------------------
        # Input:
        #
        # latent A       = 4
        # latent B       = 4
        # fused latent   = 4
        # motion masks   = 3
        #
        # Total = 15 channels
        # ----------------------------------------------------

        input_channels = (
            latent_channels * 3
            + 3
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = nn.Sequential(

            ConvBlock(
                input_channels,
                hidden_channels
            ),

            ConvBlock(
                hidden_channels,
                hidden_channels
            ),

            ConvBlock(
                hidden_channels,
                hidden_channels
            )
        )

        # ----------------------------------------------------
        # Temporal embedding
        #
        # We create one learnable embedding for each of the
        # 16 output frames.
        # ----------------------------------------------------

        self.temporal_embedding = nn.Parameter(
            torch.randn(
                num_frames,
                hidden_channels
            ) * 0.02
        )

        # ----------------------------------------------------
        # Temporal processing
        #
        # The feature map is converted into:
        #
        # [B, 16, hidden, 32, 32]
        #
        # and processed using 3D convolutions.
        # ----------------------------------------------------

        self.temporal_conv = nn.Sequential(

            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_channels
            ),

            nn.SiLU(),

            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_channels
            ),

            nn.SiLU()
        )

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        self.output = nn.Conv3d(
            hidden_channels,
            latent_channels,
            kernel_size=3,
            padding=1
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        latent_a,
        latent_b,
        fused_latent,
        background_mask,
        foreground_mask,
        residual_mask
    ):

        # ----------------------------------------------------
        # Check latent shapes
        # ----------------------------------------------------

        if latent_a.ndim != 4:
            raise ValueError(
                f"latent_a must be [B,4,32,32], "
                f"got {latent_a.shape}"
            )

        if latent_b.ndim != 4:
            raise ValueError(
                f"latent_b must be [B,4,32,32], "
                f"got {latent_b.shape}"
            )

        if fused_latent.ndim != 4:
            raise ValueError(
                f"fused_latent must be [B,4,32,32], "
                f"got {fused_latent.shape}"
            )

        # ----------------------------------------------------
        # Convert masks:
        #
        # [B,32,32]
        #
        # -> [B,1,32,32]
        # ----------------------------------------------------

        if background_mask.ndim == 3:

            background_mask = (
                background_mask.unsqueeze(1)
            )

        if foreground_mask.ndim == 3:

            foreground_mask = (
                foreground_mask.unsqueeze(1)
            )

        if residual_mask.ndim == 3:

            residual_mask = (
                residual_mask.unsqueeze(1)
            )

        # ----------------------------------------------------
        # Concatenate conditioning
        # ----------------------------------------------------

        x = torch.cat(
            [
                latent_a,
                latent_b,
                fused_latent,
                background_mask,
                foreground_mask,
                residual_mask
            ],
            dim=1
        )

        # Shape:
        #
        # [B,15,32,32]
        # ----------------------------------------------------

        features = self.encoder(x)

        # Shape:
        #
        # [B,64,32,32]
        # ----------------------------------------------------

        batch_size = features.shape[0]

        # ----------------------------------------------------
        # Create 16 temporal copies
        # ----------------------------------------------------

        features = features.unsqueeze(2)

        features = features.repeat(
            1,
            1,
            self.num_frames,
            1,
            1
        )

        # ----------------------------------------------------
        # Add temporal embeddings
        # ----------------------------------------------------

        temporal = self.temporal_embedding

        temporal = temporal.view(
            1,
            self.num_frames,
            self.hidden_channels,
            1,
            1
        )

        temporal = temporal.permute(
            0,
            2,
            1,
            3,
            4
        )

        temporal = temporal.expand(
            batch_size,
            -1,
            -1,
            32,
            32
        )

        features = features + temporal

        # ----------------------------------------------------
        # Temporal processing
        # ----------------------------------------------------

        features = self.temporal_conv(
            features
        )

        # ----------------------------------------------------
        # Predict latent frames
        # ----------------------------------------------------

        output = self.output(
            features
        )

        # output:
        #
        # [B,4,16,32,32]
        # ----------------------------------------------------

        return output


# ============================================================
# Forward-pass test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TESTING INTERPOLATIONNET")
    print("=" * 70)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = InterpolationNet(
        latent_channels=4,
        hidden_channels=64,
        num_frames=16
    ).to(device)

    model.eval()

    # --------------------------------------------------------
    # Fake batch
    #
    # This matches your real dataset shapes.
    # --------------------------------------------------------

    batch_size = 2

    latent_a = torch.randn(
        batch_size,
        4,
        32,
        32,
        device=device
    )

    latent_b = torch.randn(
        batch_size,
        4,
        32,
        32,
        device=device
    )

    fused_latent = torch.randn(
        batch_size,
        4,
        32,
        32,
        device=device
    )

    background_mask = torch.randn(
        batch_size,
        32,
        32,
        device=device
    )

    foreground_mask = torch.randn(
        batch_size,
        32,
        32,
        device=device
    )

    residual_mask = torch.randn(
        batch_size,
        32,
        32,
        device=device
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            latent_a,
            latent_b,
            fused_latent,
            background_mask,
            foreground_mask,
            residual_mask
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("Input shapes:")
    print("latent_a       :", latent_a.shape)
    print("latent_b       :", latent_b.shape)
    print("fused_latent   :", fused_latent.shape)
    print("background     :", background_mask.shape)
    print("foreground     :", foreground_mask.shape)
    print("residual       :", residual_mask.shape)

    print()
    print("Output shape:")
    print("prediction     :", prediction.shape)

    # --------------------------------------------------------
    # Expected
    # --------------------------------------------------------

    expected = (
        batch_size,
        4,
        16,
        32,
        32
    )

    if tuple(prediction.shape) != expected:

        raise RuntimeError(
            f"Wrong output shape!\n"
            f"Expected: {expected}\n"
            f"Got:      {tuple(prediction.shape)}"
        )

    print()
    print("=" * 70)
    print("FORWARD PASS PASSED")
    print("=" * 70)
