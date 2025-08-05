#!/usr/bin/env bash
exec gunicorn --bind 0.0.0.0:8008 \
              --workers 1 \
              --threads 2 \
              server:app