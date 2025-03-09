import pytest

from cornserve.task_executors.eric.schema import Modality

from ..utils import ModalityData


TEST_IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
]


@pytest.fixture(scope="session")
def test_images() -> list[ModalityData]:
    """Fixture to provide test images."""
    return [ModalityData(url=url, modality=Modality.IMAGE) for url in TEST_IMAGE_URLS]


TEST_VIDEO_URLS = [
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    "https://www.sample-videos.com/video321/mp4/360/big_buck_bunny_360p_2mb.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
    "https://www.sample-videos.com/video321/mp4/720/big_buck_bunny_720p_2mb.mp4",
]


@pytest.fixture(scope="session")
def test_videos() -> list[ModalityData]:
    """Fixture to provide test videos."""
    return [ModalityData(url=url, modality=Modality.VIDEO) for url in TEST_VIDEO_URLS]
