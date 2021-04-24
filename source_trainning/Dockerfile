FROM amazon/aws-lambda-python:3.8
RUN /var/lang/bin/python3.8 -m pip install --upgrade pip
#RUN /var/lang/bin/python3.8 pip3 install --upgrade pip
RUN pip install --upgrade setuptools

##COPY##

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY /src .

##ENTRYPOINT##
CMD [ "trainning.handler" ]

##ENTRYPOINT##
#ENTRYPOINT ["/bin/bash"]




