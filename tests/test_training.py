"""Tests for training components."""

import pytest
import torch
from torch.utils.data import DataLoader

from dimba.training.trainer import DIMBALightningModule, SimpleTrainer
from dimba.data import DummyDataset, collate_fn


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

        assert isinstance(output, tuple)
        assert len(output) == 2

    def test_configure_optimizers(self, module):
        # Need to set trainer.max_steps for LambdaLR scheduler
        class DummyTrainer:
            max_steps = 10000

        module.trainer = DummyTrainer()
        config = module.configure_optimizers()

        assert 'optimizer' in config
        assert 'lr_scheduler' in config


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
