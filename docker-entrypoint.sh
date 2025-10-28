#!/bin/sh

set -eu

cd /app

echo "[entrypoint] Ingesting documents..."
make index

echo "[entrypoint] Starting API server..."
exec make dev
