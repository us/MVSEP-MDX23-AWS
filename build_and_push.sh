#!/bin/bash

# Set image name
IMAGE_NAME="mvsep-mdx23-runpod"
TAG="latest"

echo "Building Docker image: $IMAGE_NAME:$TAG"

# Build with optimizations
DOCKER_BUILDKIT=1 docker build \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --platform linux/amd64 \
    -t $IMAGE_NAME:$TAG .

echo "Tagging image..."
docker tag $IMAGE_NAME:$TAG $USER/$IMAGE_NAME:$TAG

echo "Pushing image to Docker Hub..."
docker push $USER/$IMAGE_NAME:$TAG

echo "Done! Image pushed to Docker Hub as $USER/$IMAGE_NAME:$TAG"