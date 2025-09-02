# ~/sync_thinkerbell.sh
#!/bin/bash

SRC="/mnt/c/Users/Admin/OneDrive/Desktop/scripts/Thinkerbell/Thinkerbell_template_pipeline/"
DEST="$HOME/Thinkerbell/Thinkerbell_template_pipeline/"

rsync -av --exclude '.git' "$SRC" "$DEST"
