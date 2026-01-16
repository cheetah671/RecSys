#!/bin/bash


sudo docker stop $(sudo docker ps -q)
# sudo docker run --rm -p 3000:3000 -v $(pwd)/data:/data --tmpfs /tmp:rw,noexec,nosuid --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --memory 256m crawling_assignment:1.2

sudo docker run --rm -p 3000:3000 \
  -v $(pwd)/data:/data \
  --tmpfs /tmp:rw,noexec,nosuid \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  ire_project:1.0