"""Tests for training components."""

import pytest
import torch
from torch.utils.data import DataLoader

from dimba.training.trainer import (
    DIMBALightningModule,
    SimpleTrainer,
    VAETrainer,
    compute_consistency_loss,
    compute_dimba_losses,
)
from dimba.training.optimizers import HybridMuon
from dimba.data import DummyDataset, collate_fn
from dimba.models.embeddings import TokenEmbedding
from dimba.models.vae import TokenVAE


class TestDIMBALightningModule:
    """Test PyTorch Lightning module."""

    @pytest.fixture
    def module(self):
        return DIMBALightningModule(
            vocab_size=1000,
            model_config={
                'd_model': 64,
                'd_prompt': 64,
                'num_diffusion_steps': 100,
                'num_denoiser_layers': 2,
            },
            learning_rate=2e-5,
            warmup_steps=100,
            ema_decay=0.9999,
            use_ema=True,
        )

    def test_module_initialization(self, module):
        """Test module initializes correctly."""
        assert module.vocab_size == 1000
        assert module.use_ema
        assert module.ema_model is not None

    def test_get_model_for_inference(self, module):
        """Test getting inference model."""
        infer_model = module.get_model_for_inference()
        assert infer_model is module.ema_model

    def test_forward(self, module):
        """Test forward pass."""
        input_ids = torch.randint(0, 1000, (4, 32))
        t = torch.randint(0, 100, (4,))

        output = module(input_ids, t)

        # DIMBA.forward returns (x_pred, noise, latent_info).
        assert isinstance(output, tuple)
        assert len(output) == 3
        x_pred, noise, latent_info = output
        assert x_pred.shape == input_ids.shape + (module.model.d_model,)
        assert noise is not None
        assert isinstance(latent_info, dict)

    def test_configure_optimizers(self, module):
        # Need to set trainer.estimated_stepping_batches for the LambdaLR scheduler,
        # which computes the warmup/cosine-decay horizon from the total step count.
        class DummyTrainer:
            estimated_stepping_batches = 10000

        module.trainer = DummyTrainer()
        config = module.configure_optimizers()

        assert 'optimizer' in config
        assert 'lr_scheduler' in config

    def test_configure_muon_optimizer(self):
        module = DIMBALightningModule(
            vocab_size=100,
            model_config={
                'd_model': 32,
                'd_prompt': 32,
                'num_diffusion_steps': 10,
                'num_denoiser_layers': 1,
                'use_simple_mamba': True,
            },
            use_ema=False,
            optimizer='muon',
        )

        class DummyTrainer:
            estimated_stepping_batches = 100

        module.trainer = DummyTrainer()
        config = module.configure_optimizers()

        assert isinstance(config['optimizer'], HybridMuon)
        assert not any('token_embed' in name for name in config['optimizer'].muon_parameter_names)
        assert not any('output_head' in name for name in config['optimizer'].muon_parameter_names)


def test_consistency_loss_handles_mixed_valid_deltas_without_filter_shape_bug():
    from dimba.models.diffusion import DIMBA

    torch.manual_seed(15)
    model = DIMBA(
        vocab_size=32,
        d_model=16,
        d_prompt=16,
        num_diffusion_steps=10,
        num_denoiser_layers=1,
        d_state=4,
        use_simple_mamba=True,
        dropout=0.0,
    )
    ids = torch.randint(0, 32, (4, 6))
    loss = compute_consistency_loss(
        model,
        ids,
        model.token_embed(ids),
        torch.tensor([0, 1, 5, 9]),
        delta_min=2,
        delta_max=4,
    )

    assert torch.isfinite(loss)
    loss.backward()


def test_consistency_loss_is_zero_when_no_row_has_a_valid_delta():
    from dimba.models.diffusion import DIMBA

    model = DIMBA(
        vocab_size=16,
        d_model=8,
        d_prompt=8,
        num_diffusion_steps=8,
        num_denoiser_layers=1,
        d_state=2,
        use_simple_mamba=True,
        dropout=0.0,
    )
    ids = torch.randint(0, 16, (2, 4))
    loss = compute_consistency_loss(
        model,
        ids,
        model.token_embed(ids),
        torch.zeros(2, dtype=torch.long),
        delta_min=2,
        delta_max=4,
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss))


class TestSimpleTrainer:
    """Test simple training loop."""

    @pytest.fixture
    def trainer(self):
        from dimba.models.diffusion import DIMBA

        model = DIMBA(
            vocab_size=1000,
            d_model=64,
            num_diffusion_steps=100,
            num_denoiser_layers=2,
        )

        dataset = DummyDataset(size=100, vocab_size=1000, seq_length=32)
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=collate_fn,
        )

        return SimpleTrainer(
            model=model,
            train_dataloader=dataloader,
            val_dataloader=dataloader,
            device='cpu',
            num_epochs=1,
            learning_rate=1e-4,
        )

    def test_trainer_initialization(self, trainer):
        """Test trainer initializes correctly."""
        assert trainer.model is not None
        assert trainer.ema_model is not None
        assert trainer.optimizer is not None

    def test_copy_model_weights(self, trainer):
        """Test copying weights to EMA model."""
        trainer._copy_model_weights()

        for p1, p2 in zip(trainer.model.parameters(), trainer.ema_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_ema_update(self, trainer):
        """Test EMA update."""
        initial_ema = next(trainer.ema_model.parameters()).clone()

        # Modify main model
        with torch.no_grad():
            for p in trainer.model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        trainer._update_ema()

        updated_ema = next(trainer.ema_model.parameters())
        assert not torch.allclose(initial_ema, updated_ema)


def test_vae_trainer_flushes_partial_accumulation_and_validates() -> None:
    rows = [{"input_ids": torch.randint(0, 20, (5,))} for _ in range(5)]
    loader = DataLoader(rows, batch_size=2)
    trainer = VAETrainer(
        vae=TokenVAE(8, 4, num_layers=1, dropout=0.0),
        token_embed=TokenEmbedding(20, 8),
        train_dataloader=loader,
        val_dataloader=loader,
        device="cpu",
        num_epochs=1,
        warmup_steps=1,
        gradient_accumulation_steps=2,
    )

    trainer.train()

    assert trainer.global_step == 2
    metrics = trainer.validate()
    assert len(metrics) == 3
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_documented_mps_small_recipe_forward_backward() -> None:
    from scripts.train_interactive import PRESETS
    from dimba.models.diffusion import DIMBA

    config = dict(PRESETS["mps-small"]["model"])
    model = DIMBA(vocab_size=128, **config).to("mps")
    input_ids = torch.randint(0, 128, (1, 32), device="mps")
    timesteps = torch.randint(
        0, model.num_diffusion_steps, (1,), device="mps"
    )
    loss, _ = compute_dimba_losses(model, input_ids, timesteps)
    loss.backward()
    torch.mps.synchronize()

    assert torch.isfinite(loss.cpu())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
