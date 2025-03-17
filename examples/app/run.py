"""Invoke the app."""

import sys
import requests


def main(cornserve_url: str, app_id: str, prompt: str, image_url: str) -> None:
    """Invoke the app."""
    response = requests.post(
        f"http://{cornserve_url}/v1/apps/{app_id}",
        json={
            "request_data": {"prompt": prompt, "image_url": image_url},
        },
    )
    print(response.text)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            "Usage: python run.py <cornserve_url> <app_id> <prompt> <image_url>"
        )
        sys.exit(1)

    main(
        cornserve_url=sys.argv[1],
        app_id=sys.argv[2],
        prompt=sys.argv[3],
        image_url=sys.argv[4],
    )
