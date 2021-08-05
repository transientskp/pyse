FROM kernsuite/base:6
RUN docker-apt-install \
    python-pip \
    python-numpy \
    python-astropy \
    python-scipy \
    python-tz \
    python-casacore \
    python-dateutil \
    python-six \
    python3-pip \
    python3-numpy \
    python3-astropy \
    python3-scipy \
    python3-tz \
    python3-casacore \
    python3-dateutil \
    python3-six \
    python-coverage \
    python-nose \
    python3-nose


ADD . /code
WORKDIR /code
RUN pip install .
RUN pip3 install .
