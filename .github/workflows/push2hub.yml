name: Publish Docker image

on:
  push:
    branches: [ master, main ]

jobs:
  push_to_registry:
    name: Push Docker image to Docker Hub
    runs-on: ubuntu-latest
    steps:
      - name: Free Disk Space in Ubuntu
        uses: jlumbroso/free-disk-space@main
        with:
          tool-cache: true
          android: true
          dotnet: true
          haskell: true
          large-packages: false
          docker-images: true
          swap-storage: true

      - name: Check out the repo
        uses: actions/checkout@v3
        with:
          fetch-depth: 2

      - name: Check for VERSION change in .env
        id: check_version
        run: |
          if git diff HEAD^ --name-only | grep .env; then
            echo "::set-output name=version_changed::true"
          else
            echo "::set-output name=version_changed::false"
          fi

      - name: Log in to Docker Hub
        if: steps.check_version.outputs.version_changed == 'true'
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
          logout: true

      - name: Build and push Docker images
        if: steps.check_version.outputs.version_changed == 'true'
        run: |
          if [ -f ".env" ] && [ -f "Dockerfile" ]; then
            IMAGE_VERSION=$(grep -oP '(?<=VERSION = )\S+' .env)
            REPO_NAME=$(echo $GITHUB_REPOSITORY | awk -F / '{print tolower($2)}')
            IMAGE_NAME=${{ secrets.DOCKER_USERNAME }}/$REPO_NAME:$IMAGE_VERSION
            docker build -t $IMAGE_NAME -f Dockerfile .
            docker push $IMAGE_NAME
          fi

