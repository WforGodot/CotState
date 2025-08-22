.PHONY: upload-activations rclone-auth

# Upload activations to Google Drive using rclone.
# Defaults:
#   SOURCE_DIR=src/cot/activations_outputs
#   REMOTE_NAME=gdrive
#   REMOTE_PATH=cot-activations
# Override via environment variables when invoking make.

upload-activations:
	@echo "[make] Uploading files via rclone..."
	@bash scripts/sync_files_to_gdrive.sh

# Run rclone config headless (if you need to re-auth)
rclone-auth:
	@rclone config

