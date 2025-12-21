"""Tests for c2i2o.core.cosmology module."""

import pytest

from c2i2o.core.cosmology import CosmologyBase


class TestCosmologyBase:
    """Tests for CosmologyBase abstract class."""

    def test_cannot_instantiate(self) -> None:
        """Test that CosmologyBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CosmologyBase(cosmology_type="test")  # type: ignore
