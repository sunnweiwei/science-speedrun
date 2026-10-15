"""Thin CLI compatibility for OMatG's unchanged unconditional sampler."""

from typing import Optional

from omg.sampler.abstracts import CellDistribution, PositionDistribution, SpeciesDistribution
from omg.sampler.independent_sampler import IndependentSampler


class UnconditionalIndependentSampler(IndependentSampler):
    """Accept OMGCLI's linked batch size without changing base-state sampling."""

    def __init__(
        self,
        species_distribution: SpeciesDistribution,
        pos_distribution: PositionDistribution,
        cell_distribution: CellDistribution,
        batch_size: Optional[int] = None,
    ):
        super().__init__(species_distribution, pos_distribution, cell_distribution)
        self.batch_size = batch_size
