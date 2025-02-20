import pytest

from cornserve.task_executors.eric.schema import Modality

from ..utils import TestModalityData


@pytest.fixture(scope="session")
def large_test_image() -> TestModalityData:
    """Fixture to provide a large test image."""
    return TestModalityData(
        url="https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
        modality=Modality.IMAGE,
    )
