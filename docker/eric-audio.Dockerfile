# We have a seaprate Dockerfile due to license issues with the audio package.
# Users who need audio support in Eric should read and accept their license terms,
# build this image separately, and tag it as `cornserve/eric` so that it's picked up.
FROM cornserve/eric:latest

RUN pip install -e '.[audio]'
