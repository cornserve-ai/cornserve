# Task Registry

This submodule is in charge of managing task classes (unit, composite, descriptors) and concreate task instances.

- The `registry.py` includes all the utilities for k8s Custom Resource (CR) interactions, including creating/watching/getting the classes and task instances.
- The `task_class_registry` and `descriptor_registry.py` are in charge of loading and fetching task classes into the python runtime of each service process.
