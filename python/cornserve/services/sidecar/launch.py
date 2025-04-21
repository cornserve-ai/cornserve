"""Luaunch information for sidecars."""

from __future__ import annotations

import kubernetes_asyncio.client as kclient

from cornserve import constants


class SidecarLaunchInfo:
    """Information to launch sidecars."""

    @classmethod
    def get_pod(cls, node: kclient.V1Node, sidecar_rank: int, world_size: int) -> kclient.V1Pod:
        """Get the pod spec for the sidecar.

        Args:
            node: The Kubernetes node to launch the sidecar on.
            sidecar_rank: The global rank of the sidecar.
            world_size: The total number of sidecars in the cluster.
        """
        if not node.metadata:
            raise ValueError("Node metadata is missing")
        pod_name = f"sidecar-{sidecar_rank}"
        return kclient.V1Pod(
            metadata=kclient.V1ObjectMeta(
                name=pod_name,
                labels={
                    "app": "sidecar",
                    "sidecar-rank": str(sidecar_rank),
                },
            ),
            spec=kclient.V1PodSpec(
                containers=[
                    kclient.V1Container(
                        name="sidecar",
                        image=constants.CONTAINER_IMAGE_SIDECAR,
                        image_pull_policy="Always",
                        ports=[
                            kclient.V1ContainerPort(
                                container_port=constants.K8S_SIDECAR_SERVICE_PORT,
                                name=constants.K8S_SIDECAR_SERVICE_PORT_NAME,
                            ),
                        ],
                        env=[
                            kclient.V1EnvVar(name=name, value=value)
                            for name, value in cls.get_envs(
                                sidecar_rank,
                                world_size,
                            )
                        ],
                        volume_mounts=[
                            kclient.V1VolumeMount(
                                name=name,
                                mount_path=container_path,
                            )
                            for name, _, container_path in cls.get_container_volumes()
                        ],
                    )
                ],
                volumes=[
                    kclient.V1Volume(
                        name=name,
                        host_path=kclient.V1HostPathVolumeSource(path=host_path),
                    )
                    for name, host_path, _ in cls.get_container_volumes()
                ],
                service_account_name="sidecar",
                runtime_class_name="nvidia",
                node_name=node.metadata.name,
                host_ipc=True,
                host_pid=True,
                hostname=f"sidecar-{sidecar_rank}",
                subdomain="sidecar",
            ),
        )

    @staticmethod
    def get_envs(sidecar_rank: int, world_size: int) -> list[tuple[str, str]]:
        """Get the environment variables for the sidecar.

        Args:
            sidecar_rank: The global rank of the sidecar.
            world_size: The total number of sidecars in the cluster.
        """
        return [
            (
                "SIDECAR_MASTER_ADDR",
                f"{constants.K8S_SIDECAR_SERVICE_NAME}-0.{constants.K8S_SIDECAR_SERVICE_NAME}.{constants.K8S_NAMESPACE}.svc.cluster.local",
            ),
            ("SIDECAR_MASTER_PORT", str(constants.K8S_SIDECAR_SERVICE_PORT)),
            ("SIDECAR_WORLD_SIZE", str(world_size)),
            ("SIDECAR_POD_NAME", str(f"sidecar-{sidecar_rank}")),
        ]

    @staticmethod
    def get_container_image() -> str:
        """Get the container image for the sidecar."""
        return constants.CONTAINER_IMAGE_SIDECAR

    @staticmethod
    def get_container_volumes() -> list[tuple[str, str, str]]:
        """Get the container volumes for the sidecar."""
        return [("shm-volume", constants.VOLUME_SHM, "/dev/shm")]
