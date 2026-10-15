#!/usr/bin/env bash
# Dừng tất cả container Docker của dự án để giải phóng RAM & CPU
set -e
cd "$(dirname "$0")"

echo "Đang dừng các container Milvus, MinIO, etcd..."
if docker ps >/dev/null 2>&1; then
    docker compose stop
else
    sg docker -c "docker compose stop"
fi

echo "✅ Đã dừng toàn bộ container. RAM & CPU đã được giải phóng!"
