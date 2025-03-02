import pytest

from cornserve.task_executors.eric.schema import Modality

from ..utils import ModalityData


@pytest.fixture(scope="session")
def test_images() -> list[ModalityData]:
    """Fixture to provide test images."""
    return [
        ModalityData(
            url="https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
            modality=Modality.IMAGE,
        ),
        ModalityData(
            url="https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
            modality=Modality.IMAGE,
        ),
    ]
