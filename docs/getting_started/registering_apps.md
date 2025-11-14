# Deploying Apps to Cornserve and Invoking Them

Once you've written your app, you can deploy it to Cornserve.
The current deployment process is as follows:

1. Save the app code in a single Python file (e.g., `gemmarena.py`).
2. If you haven't already, deploy the Cornserve tasklib to the cluster.
    ```bash
    export CORNSERVE_GATEWAY_URL=[...]
    cornserve tasklib-deploy
    ```
3. Register & deploy the app to the Cornserve Gateway for validation and deployment:
    ```bash
    cornserve register gemmarena.py
    ```
4. When validation succeeds, the Cornserve Gateway will deploy the app and all its subtasks on the Cornserve data plane, and the `cornserve` CLI invocation will return with the app's ID.
5. Finally, you can invoke the app using the Cornserve CLI or send requests to the Cornserve Gateway with your choice of HTTP client.

## Using the Cornserve CLI

For streaming responses like our Gemma Arena example, you can use the CLI to invoke the app:

```bash
cornserve invoke gemmarena --aggregate-keys gemma3-4b gemma3-12b gemma3-27b --data - <<EOF
model: gemmas
messages:
- role: "user"
  content:
  - type: text
    text: "Write a poem about the images you see."
  - type: image_url
    image_url:
      url: "https://picsum.photos/id/12/480/560"
  - type: image_url
    image_url:
      url: "https://picsum.photos/id/234/960/960"
EOF
```

Notice that this is basically a YAML representation of `OpenAIChatCompletionRequest`.

## Next Steps

To dive deeper into the architecture of Cornserve, check out our [architecture guide](../architecture/index.md).
