"""Tests for sea_mosaic.config.PipelineConfig (fields for the Stage A->D pipeline)."""

from __future__ import annotations

import dataclasses

import pytest

from sea_mosaic.config import PipelineConfig


def test_config_needs_no_arguments_and_defaults_to_gps_heading_anchors() -> None:
    config = PipelineConfig()
    assert config.heading_anchor_source == "gps"
    assert config.latlons is None
    assert config.gimbal_yaw_deg is None
    assert config.pairs is None and config.loops is None
    assert config.ransac_threshold == 3.0


def test_config_fields_are_exactly_the_new_set() -> None:
    assert [f.name for f in dataclasses.fields(PipelineConfig)] == [
        "ransac_threshold",
        "output_dir",
        "latlons",
        "heading_anchor_source",
        "gimbal_yaw_deg",
        "pairs",
        "loops",
    ]


@pytest.mark.parametrize(
    "old_field",
    ["altitude_m", "dfov_deg", "reference_index", "gps_positions", "gimbal_yaw", "yaw_anchor_weight"],
)
def test_removed_joint_optimization_fields_are_rejected(old_field: str) -> None:
    """A caller still passing an old field must fail loudly, not have it silently dropped."""
    with pytest.raises(TypeError):
        PipelineConfig(**{old_field: 1.0})
