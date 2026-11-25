#!/usr/bin/env bash
# Rebuild everything from scratch (about 15 minutes on a laptop)
set -e
python -m src.fetch_data
python -m src.build_dataset
python -m src.train
python -m src.make_art
python -m src.export_web
echo "Done. Start the app with:  uvicorn api.main:app --port 8000   then open http://localhost:8000"
