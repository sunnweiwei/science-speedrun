import torch

from reference.alphagenome_50m.model import (
    CompactModel,
    ModelConfig,
    count_parameters,
)


def test_default_model_stays_below_parameter_limit():
    assert count_parameters(CompactModel()) <= 50_000_000


def test_multiresolution_output_shapes():
    config = ModelConfig(
        input_length=8_192,
        target_length=2_048,
        trunk_bin_size=256,
        contact_bin_size=512,
        stem_stride=8,
        encoder_width=8,
        channel_growth=4,
        num_downsampling_blocks=5,
        hidden_dim=16,
        num_sequence_blocks=1,
        num_attention_heads=2,
        attention_head_dim=8,
        mlp_expansion=2,
        head_hidden_size=12,
        junction_rank=4,
        gradient_checkpointing=False,
    )
    model = CompactModel(config).eval()
    dna = torch.zeros(1, config.input_length, 4)
    dna[:, :, 0] = 1
    usage_positions = torch.tensor([[0, 127, 1_024, 2_047]])
    site_positions = torch.arange(512).repeat(1, 4, 1) % config.target_length
    queries = torch.tensor([[[0, 1, 2], [510, 511, 733]]])
    with torch.no_grad():
        output = model(
            dna,
            usage_positions=usage_positions,
            junction_site_positions=site_positions,
            junction_queries=queries,
        )
    assert output["atac"].shape == (1, 16, 167)
    assert output["chip_tf"].shape == (1, 16, 1_617)
    assert output["splice_sites"].shape == (1, 2_048, 4)
    assert output["splice_usage"].shape == (1, 4, 734)
    assert output["junctions"].shape == (1, 2)
    assert output["contact_maps"].shape == (1, 136, 28)


def test_predicted_position_and_junction_block_shapes():
    config = ModelConfig(
        input_length=8_192, target_length=2_048, trunk_bin_size=256,
        contact_bin_size=512, stem_stride=8, encoder_width=8,
        channel_growth=4, num_downsampling_blocks=5, hidden_dim=16,
        num_sequence_blocks=1, num_attention_heads=2, attention_head_dim=8,
        mlp_expansion=2, head_hidden_size=12, junction_rank=4,
        gradient_checkpointing=False,
    )
    model = CompactModel(config).eval()
    dna = torch.zeros(1, config.input_length, 4)
    with torch.no_grad():
        trunk = model._trunk(dna)
        logits = model._splice_site_logits(trunk)
        positions = model.select_junction_positions(logits)
        block = model.junction_block(trunk, positions, 0, 3, 4, 6, 365, 369)
        cached = model.junction_block(
            trunk, positions, 0, 3, 4, 6, 365, 369,
            model.junction_site_factors(trunk, positions),
        )
    assert logits.shape == (1, 8_192, 4)
    assert positions.shape == (1, 4, 512)
    assert torch.all(positions[:, :, 1:] >= positions[:, :, :-1])
    assert block.shape == (1, 3, 2, 4)
    assert torch.equal(block, cached)
