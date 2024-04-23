FROM ubuntu:latest

#Set timezone
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /lookielookie

COPY requirements.txt .
COPY setup.py .

RUN apt-get -y update \
    && apt-get install -yq cron nano python3 python3-pip tzdata \
    && touch /var/log/cron.log \
    && pip3 install -r requirements.txt

COPY /lookielookie /lookielookie/lookielookie
RUN pip3 install .
COPY crontab /etc/cron.d/cjob
RUN chmod 0644 /etc/cron.d/cjob
RUN chmod 0644 ./lookielookie/start.sh && chmod +x ./lookielookie/start.sh && chmod +x ./lookielookie/job.sh && chmod +x ./lookielookie/job_fundamentals.sh

CMD ["/lookielookie/lookielookie/start.sh"]