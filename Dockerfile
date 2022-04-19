FROM kernsuite/base:6
RUN docker-apt-install \
    python3-pip \
    python3-numpy \
    python3-astropy \
    python3-scipy \
    python3-tz \
    python3-casacore \
    python3-dateutil \
    python3-six \
    python3-nose \
    python3-psutil

ADD . /code
WORKDIR /code
RUN pip3 install --upgrade pip
RUN pip3 install coverage dask[complete] ray
RUN pip3 install .
