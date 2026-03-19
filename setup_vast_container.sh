#!/usr/bin/env bash
set -euo pipefail
SETUP_DATA=1

mkdir -p "${DATA_DIRECTORY:-/root/}" /root/data "$HOME/.config/rclone"
mkdir -p /root/repo
mkdir -p /root/data/trained-mega-models
cd /root

if [ "$SETUP_DATA" -eq 1 ]; then
  apt-get install -y rclone
  cat >"$HOME/.config/rclone/rclone.conf"<<'EOF'
[gdrive]
type = drive
scope = drive

# Add client_id, client_secret and token below


EOF
  chmod 600 "$HOME/.config/rclone/rclone.conf"

  ( cd /root/data && rclone copy -P --transfers 2 \
      gdrive:"megatron_dir/mega-models/Qwen3-0.6B" \
      mega-models/Qwen3-0.6B ) &

  ( cd /root/data && rclone copy -P --transfers 2 \
      gdrive:"megatron_dir/hf_models/Qwen3-0.6B" \
      hf_models/Qwen3-0.6B ) &

  ( cd /root/data && rclone copy -P --transfers 16 --checkers 32 --fast-list --buffer-size 128M \
      gdrive:"megatron_dir/datasets/" datasets/ ) &
fi

( cd /root/repo/ && { [ -d slime ] || git clone https://github.com/orangeaico/slime.git; } && \
  cd slime && git checkout swe_agent_integration && \
  bash /root/repo/slime/setup_slime_container.sh ) &

for pid in $(jobs -p); do wait "$pid" || exit 1; done

cd /root/repo/slime && echo "All Done!"

