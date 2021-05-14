#!/usr/bin/env bash

set -exo pipefail

function slack()
{
  curl -X POST -H 'Content-type: application/json' --data "{\"text\":\"$1\"}" https://hooks.slack.com/services/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
}

echo "$(date): NEW_ENCODING"

cd "$(dirname $0)/.."

PATH_TO_JSON_FILE=$1

echo "PATH_TO_JSON_FILE: $PATH_TO_JSON_FILE"

. venv/bin/activate

ENCODING_COMMAND="python jukebox/get_encodings.py $PATH_TO_JSON_FILE"
echo "$(date): ENCODING_START"
slack "ENCODING_START"
bash -c "$ENCODING_COMMAND"
echo "$(date): ENCODING_COMPLETE"
slack "ENCODING_COMPLETE: $ENCODING_COMMAND"

echo "$(date): DONE"
