# Deploying Apps to Cornserve and Invoking Them

Once you've written your app, you can deploy it to Cornserve.
The current deployment process is as follows:

1. Save the app code in a single Python file (e.g., `image_chat.py`).
2. Register & deploy the app to the Cornserve Gateway for validation and deployment:
    ```bash
    export CORNSERVE_GATEWAY_URL=[...]
    cornserve register image_chat.py
    ```
3. When validation succeeds, the Cornserve Gateway will deploy the app and all its subtasks on the Cornserve data plane, and the `cornserve` CLI invocation will return with the app's ID.
4. Finally, you can send requests to the Cornserve Gateway with your choice of HTTP client.
    ```python
    response = requests.post(
        f"{CORNSERVE_GATEWAY_URL}/app/invoke/{APP_ID}",
        json={
            "request_data": {
                "image_url": "https://example.com/image.jpg",
                "prompt": "Describe the image.",
            }
        },
    )
    ```
    Notice that what comes within the `"request_data"` key is the JSON representation of your `Request` class defined in our [previous example](building_apps.md#app).

## Next Steps

To dive deeper into the architecture of Cornserve, check out our [architecture guide](../architecture/index.md).

