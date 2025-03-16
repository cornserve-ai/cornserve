"""Register vlm.py with Cornserve."""

import sys
import requests


def main(cornserve_url: str, source_path: str) -> None:
    """Register the app with Cornserve."""
    with open(source_path) as f:
        source_code = f.read()
    response = requests.post(
        f"http://{cornserve_url}/admin/register_app",
        json={"source_code": source_code},
    )
    print(response.text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python register_app.py <cornserve_url> [<source_path>]")
        sys.exit(1)

    main(
        cornserve_url=sys.argv[1],
        source_path=sys.argv[2] if len(sys.argv) > 2 else "vlm.py",
    )
