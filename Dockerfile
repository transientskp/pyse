FROM kernsuite/base:3
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
    python3-nose \
    python-psutil \
    python3-psutil \
    python-ray \
    python3-ray


ADD . /code
WORKDIR /code
RUN pip install .
RUN pip3 install .
