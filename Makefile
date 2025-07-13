BUILD_HOST=192.168.0.13:32000
NAME=feedoscope
TAG=$(shell git log -1 --pretty=%h)

train:
	LOGGING_CONFIG=dev_logging.conf uv run python -m feedoscope.pu_learn

pkg:
	echo "Tag: " ${TAG}
	echo ${NAME}:${TAG}
	docker build -t ${NAME}:${TAG} .
	-docker inspect "${NAME}:latest" > /dev/null 2>&1 || docker rmi ${NAME}:latest
	docker tag ${NAME}:${TAG} ${NAME}:latest
	docker tag ${NAME}:${TAG} ${BUILD_HOST}/${NAME}:${TAG}
	docker tag ${NAME}:${TAG} ${BUILD_HOST}/${NAME}:latest
	# docker push ${BUILD_HOST}/${NAME}:${TAG}
	# docker push ${BUILD_HOST}/${NAME}:latest
	# trivy image ${NAME}:${TAG}

install:
	uv sync

infer:
	LOGGING_CONFIG=dev_logging.conf uv run python -m feedoscope.infer
