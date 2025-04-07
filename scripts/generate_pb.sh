#!/usr/local/env bash

set -evo pipefail

PROTO_DIR="proto/v1"
PROTO_FILES=$(find $PROTO_DIR -name "*.proto")

PYTHON_OUTPUT_DIR="python/cornserve/services/pb"
mkdir -p "$PYTHON_OUTPUT_DIR"
rm "$PYTHON_OUTPUT_DIR"/*pb2.py "$PYTHON_OUTPUT_DIR"/*pb2.pyi "$PYTHON_OUTPUT_DIR"/*pb2_grpc.py || true
for proto in $PROTO_FILES; do
  echo "Generating code for $proto in $PYTHON_OUTPUT_DIR"
  python -m grpc_tools.protoc \
    -I$PROTO_DIR \
    --python_out=$PYTHON_OUTPUT_DIR \
    --grpc_python_out=$PYTHON_OUTPUT_DIR \
    --pyi_out=$PYTHON_OUTPUT_DIR \
    $proto
done

# The generated `import common_pb2`, for example, doesn't work.
# We need to change it manually to `from . import common_pb2`.
# `-i.bak` is done for GNU and BSD sed compatibility.
find "$PYTHON_OUTPUT_DIR" -type f -name "*.py" -exec sed -i.bak -e 's/^\(import.*pb2\)/from . \1/g' {} \;
find "$PYTHON_OUTPUT_DIR" -type f -name "*.bak" -exec rm {} \;
