import os
import numpy as np
import cv2
import mimetypes
import base64

def _file_to_data_uri(path: str) -> str:
    """
    Read a binary file and return a data-URI, e.g.
        data:image/png;base64,iVBORw0KGgo…
    """
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    with open(path, "rb") as f:
        payload = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{payload}"

def get_image_data_uris(filenames: list[str], root: str = "images") -> list[str]:
    """Return list of data-URIs for image files."""
    return [_file_to_data_uri(os.path.join(root, name)) for name in filenames]

def create_dummy_image(
    height: int,
    width: int,
    id: int = 0,
) -> str:
    """Create a dummy image with a fixed color based on the id."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    if height > width:
        height, width = width, height

    filename = f"{height}x{width}_{id}.png"
    os.makedirs("images", exist_ok=True)
    path = os.path.join("images", filename)
    if os.path.exists(path):
        return filename

    # Deterministic BGR color from id
    b = 32 + (id * 73) % 192
    g = 32 + (id * 151) % 192
    r = 32 + (id * 191) % 192
    img = np.full((height, width, 3), (b, g, r), dtype=np.uint8)

    # Fixed PNG compression for consistent output; 0 is fastest
    if not cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise RuntimeError(f"Failed to save image: {path}")

    return filename

