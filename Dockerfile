FROM tensorflow/tensorflow:nightly-py3

RUN pip3 install scipy

RUN pip3 install pandas

RUN pip3 install scikit-learn

RUN pip3 install opencv-python

RUN apt update && apt install -y libsm6 libxext6 libxrender-dev

RUN mkdir -p /train/

RUN mkdir -p /test/

RUN mkdir -p /output/

COPY train /train

COPY test /test

COPY main.py /usr/local/bin/

COPY run.sh /run.sh  

RUN chmod a+x /usr/local/bin/main.py

RUN chmod a+x /run.sh

ENTRYPOINT ["/bin/bash", "/run.sh"]
