"""The Task Dispatcher is the interface between App Drivers and the data plane.

It receives requests from App Drivers and dispatches them to the appropriate
task manager. It also receives updates from the resource manager about changes
in Task Managers.

The Task Dispatcher requires both a gRPC server and a REST HTTP server.
The gRPC server is used by the Resource Manager to send updates about
Task Manager. On the other hand, the REST API server is used by App Drivers
to invoke tasks.
"""
