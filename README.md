<div align="center">
<h1>Cornserve: <i>Easy, Fast, and Scalable Multimodal AI</i></h1>

[![Docker Hub](https://custom-icon-badges.demolab.com/badge/Docker-cornserve-1D63ED.svg?logo=docker&logoColor=white)](https://hub.docker.com/r/cornserve/gateway)
[![Homepage](https://custom-icon-badges.demolab.com/badge/Docs-cornserve.ai-dddddd.svg?logo=home&logoColor=white&logoSource=feather)](https://cornserve.ai/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.12118)
[![Apache-2.0 License](https://custom-icon-badges.herokuapp.com/github/license/cornserve-ai/cornserve?logo=law)](/LICENSE)
</div>


https://github.com/user-attachments/assets/583796e8-da20-4918-9311-db05e94726eb

---
**Project News**  

- \[2026/04/15\] Our paper ("Cornserve: A Distributed Serving System for Any-to-Any Multimodal Models") has been accepted to ACM CAIS 2026! [Paper](https://arxiv.org/abs/2603.12118)
- \[2025/12/16\] Our planner paper preprint ("Cornfigurator: Automated Planning for Any-to-Any Multimodal Model Serving") is now available on [arXiv](https://arxiv.org/abs/2512.14098).
- \[2025/11/14\] Cornserve project announced with v0.1.0 release!
---



Cornserve is a distributed inference platform for complex, Any-to-Any multimodal AI.
Split complex models into smaller separately scalable components (**model fission**) and share common components across multiple applications (**sharing**), all on your own infrastructure.

See also [Cornfigurator](https://github.com/cornserve-ai/cornfigurator), an automated deployment planner for Cornserve.

## Getting Started

You can quickly try out Cornserve on top of Minikube. Check out our [getting started guide](https://cornserve.ai/getting_started/)!

Cornserve can be deployed on Kubernetes with a single command. More on our [docs](https://cornserve.ai/getting_started/).


## Research

These research papers describe Cornserve's system architecture and planner.

1. Cornserve (CAIS 26): [Paper](https://arxiv.org/abs/2603.12118)
1. Cornfigurator: [Paper](https://arxiv.org/abs/2512.14098) | [Repository](https://github.com/cornserve-ai/cornfigurator)

If you find Cornserve relevant to your research, please consider citing:

```bibtex
@inproceedings{cornserve-cais26,
    title     = {Cornserve: A Distributed Serving System for Any-to-Any Multimodal Models},
    author    = {Chung, Jae-Won and Ma, Jeff J. and Ahn, Jisang and Liang, Yizhuo and Jajoo, Akshay and Lee, Myungjin and Chowdhury, Mosharaf},
    booktitle = {ACM CAIS},
    year      = {2026}
}

@article{cornfigurator-arxiv25,
    title   = {Cornfigurator: Automated Planning for Any-to-Any Multimodal Model Serving},
    author  = {Ma, Jeff J. and Chung, Jae-Won and Ahn, Jisang and Liang, Yizhuo and Lu, Runyu and Jajoo, Akshay and Lee, Myungjin and Chowdhury, Mosharaf},
    journal = {arXiv preprint arXiv:2512.14098},
    year    = {2025}
}
```


## Contributing

Cornserve is an open-source project, and we welcome contributions!
Please check out our [contributor guide](https://cornserve.ai/contributor_guide/) for more information on how to get started.
