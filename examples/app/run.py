"""Invoke the app."""

import sys
import json
import requests


def main(cornserve_url: str, app_id: str, prompt: str, image_urls: list[str]) -> None:
    """Invoke the app."""
    data = {"request_data": {"prompt": prompt, "image_urls": image_urls}}

    response = requests.post(
        f"http://{cornserve_url}/v1/apps/{app_id}",
        json=data,
    )

    resp = json.loads(response.text)
    print(eval(resp["text"]))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python run.py <cornserve_url> <app_id> <prompt> [<image_url> ..]"
        )
        sys.exit(1)

    main(
        cornserve_url=sys.argv[1],
        app_id=sys.argv[2],
        prompt=sys.argv[3],
        image_urls=sys.argv[4:],
    )
