"""Unit tests for the unified ``MissingExtraError`` contract.

Verifies:
- ``MissingExtraError`` carries the structured fields.
- ``for_extra`` builds a properly-shaped instance.
- Subsystem subclasses (``MultimodalMissingExtraError``,
  ``UrlFetchMissingExtraError``) preserve backwards-compat catches.
"""

from __future__ import annotations

import pytest

from z3rno_core.extras import MissingExtraError
from z3rno_core.loaders.url import UrlFetchError, UrlFetchMissingExtraError
from z3rno_core.multimodal.base import (
    MultimodalError,
    MultimodalMissingExtraError,
    MultimodalProviderError,
)


class TestMissingExtraError:
    def test_structured_fields(self) -> None:
        err = MissingExtraError(
            "Pillow missing",
            extra_name="multimodal-local",
            dependency="Pillow",
            install_command="pip install ...",
        )
        assert err.extra_name == "multimodal-local"
        assert err.dependency == "Pillow"
        assert err.install_command == "pip install ..."
        assert str(err) == "Pillow missing"

    def test_for_extra_builds_default_install_command(self) -> None:
        err = MissingExtraError.for_extra(
            extra_name="multimodal-local",
            dependency="sentence-transformers",
        )
        assert "z3rno-core[multimodal-local]" in err.install_command
        assert "sentence-transformers" in str(err)
        assert "multimodal-local" in str(err)

    def test_for_extra_with_action(self) -> None:
        err = MissingExtraError.for_extra(
            extra_name="playwright",
            dependency="playwright",
            action="and run `playwright install chromium` once.",
        )
        assert "playwright install chromium" in str(err)


class TestMultimodalMissingExtra:
    def test_caught_by_multimodal_provider_error(self) -> None:
        """Backwards compat: ``except MultimodalProviderError`` still catches."""
        with pytest.raises(MultimodalProviderError):
            raise MultimodalMissingExtraError.for_extra(
                extra_name="multimodal-local",
                dependency="Pillow",
            )

    def test_caught_by_multimodal_error_base(self) -> None:
        with pytest.raises(MultimodalError):
            raise MultimodalMissingExtraError.for_extra(
                extra_name="multimodal-local",
                dependency="Pillow",
            )

    def test_caught_by_missing_extra_error(self) -> None:
        """The new uniform handle: ``except MissingExtraError`` catches."""
        with pytest.raises(MissingExtraError):
            raise MultimodalMissingExtraError.for_extra(
                extra_name="multimodal-local",
                dependency="Pillow",
            )

    def test_carries_structured_fields(self) -> None:
        err = MultimodalMissingExtraError.for_extra(
            extra_name="multimodal-local",
            dependency="sentence-transformers",
        )
        assert err.extra_name == "multimodal-local"
        assert err.dependency == "sentence-transformers"


class TestUrlFetchMissingExtra:
    def test_caught_by_url_fetch_error(self) -> None:
        with pytest.raises(UrlFetchError):
            raise UrlFetchMissingExtraError.for_extra(
                extra_name="playwright",
                dependency="playwright",
            )

    def test_caught_by_missing_extra_error(self) -> None:
        with pytest.raises(MissingExtraError):
            raise UrlFetchMissingExtraError.for_extra(
                extra_name="playwright",
                dependency="playwright",
            )

    def test_carries_structured_fields(self) -> None:
        err = UrlFetchMissingExtraError.for_extra(
            extra_name="playwright",
            dependency="playwright",
            action="and run `playwright install chromium` once.",
        )
        assert err.extra_name == "playwright"
        assert err.dependency == "playwright"
        assert "chromium" in str(err)
