---
description: "Easy, fast, and scalable multimodal AI"
hide:
  - navigation
  - toc
---

<div align="center">
  <h1 style="font-size: 2rem; margin-bottom: 0px">Cornserve: Easy, Fast, and Scalable Multimodal AI</h1>
</div>

<div align="center" class="video-container">
  <video src="assets/video/cornserve.mp4" id="video" 
     muted
     autoplay
     playsinline
     webkit-playsinline="true" 
     x-webkit-airplay="true"
     alt="type:video"
     type="video/mp4">
    </video>
</div>

<script>
// snippets to fix video in WeChat Browsers
// for IOS
document.addEventListener('WeixinJSBridgeReady', function() {
  const video = document.getElementById('video');
  video.play();
});
// for Andriod
document.addEventListener('DOMContentLoaded', function() {
  const userAgent = navigator.userAgent;
  if (userAgent.includes('WeChat') && /Android/.test(userAgent)) {
    const video = document.getElementById('video');
    if (video) {
      video.setAttribute('controls', 'controls');
    }
  }
});

</script>

Cornserve is an execution platform for multimodal AI.
It performs **model fission** and **automatic sharing** of common components across applications on your infrastructure.

<div class="grid cards" markdown>

-   :material-vector-intersection:{ .lg .middle } **Model fission**

    ---

    Split up your complex models into smaller components and
    scale them independently.

-   :material-share-variant:{ .lg .middle }  **Automatic sharing**

    ---

    Common model components are automatically shared across applications.

-   :material-hub:{ .lg .middle } **Multimodal-native**

    ---

    Cornserve is built multimodal-native from the ground up. Image, video,
    audio, and text are all first-class citizens.

-   :material-kubernetes:{ .lg .middle } **Deploy to K8s with one command**

    ---

    One-command deployment to Kubernetes with [Kustomize](https://kustomize.io/).

-   :simple-opentelemetry:{ .lg .middle } **Observability**

    ---

    Built-in support for [OpenTelemetry](https://opentelemetry.io/)
    to monitor your apps and requests.

-   :material-scale-balance:{ .lg .middle } **Open Source, Apache-2.0**

    ---

    Cornserve is open-source with the Apache 2.0 license and is available on
    [GitHub](https://github.com/cornserve-ai/cornserve).

</div>
