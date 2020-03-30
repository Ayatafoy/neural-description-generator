FROM python:3.7-buster
WORKDIR /workspace
COPY . /workspace
RUN pip install mkl
RUN python setup.py install
CMD ls
