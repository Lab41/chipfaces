# chipfaces
search images for faces, box bound face and write to file.

# build
docker build -f Dockerfile -t chipfaces .

# run
docker run -v /inFileDir:/in -v /outFileDir:/out chipfaces
