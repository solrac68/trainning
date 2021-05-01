FROM amazon/aws-lambda-python:3.8
RUN /var/lang/bin/python3.8 -m pip install --upgrade pip
RUN pip install --upgrade setuptools

##COPY##

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY /src .

##ENTRYPOINT##
CMD [ "opinion.handler" ]

##ENTRYPOINT##
#ENTRYPOINT ["/bin/bash"]




