echo "Applying fp32 lm head and cce patch to Megatron LM"

bash /root/repo/slime/docker/apply_megatron_fp32_lm_head_patch.sh
bash /root/repo/slime/docker/apply_megatron_cce_patch.sh

echo "fp32 and cce application done"