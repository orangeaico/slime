#!/bin/bash
# Monitor Docker containers during SWE-bench training
# Usage: bash monitor_docker.sh

echo "=== Docker Container Monitor for SWE-bench ===="
echo "Starting monitoring... Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "=== Docker Containers at $(date) ==="
    echo ""

    # Show all containers (running and stopped)
    echo "Running containers:"
    docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}\t{{.Ports}}" | head -20

    echo ""
    echo "Recently exited containers:"
    docker ps -a --filter "status=exited" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}" | head -10

    echo ""
    echo "Container count:"
    echo "  Running: $(docker ps -q | wc -l)"
    echo "  Total:   $(docker ps -a -q | wc -l)"

    echo ""
    echo "Recent logs from SWE containers (if any):"
    # Find containers with swe or python in name/image
    SWE_CONTAINERS=$(docker ps --format "{{.ID}}" --filter "name=swe" 2>/dev/null || docker ps --format "{{.ID}}" --filter "ancestor=python" 2>/dev/null | head -1)
    if [ ! -z "$SWE_CONTAINERS" ]; then
        for container_id in $SWE_CONTAINERS; do
            echo "--- Container $container_id logs (last 5 lines) ---"
            docker logs --tail 5 "$container_id" 2>&1 | head -20
        done
    else
        echo "  No SWE-agent containers found"
    fi

    echo ""
    echo "Refreshing in 3 seconds... (Ctrl+C to exit)"
    sleep 3
done
