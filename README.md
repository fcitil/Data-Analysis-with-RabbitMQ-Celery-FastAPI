## Table of Contents

- [Up and Run](#up-and-run)
- [Trigger a task](#trigger-a-task)

### Up and Run

```sh
git clone https://github.com/fcitil/Data-Analysis-with-RabbitMQ-Celery-FastAPI.git
cd Data-Analysis-with-RabbitMQ-Celery-FastAPI
docker-compose up -d
```

You'll be able to see the backend project running at [http://localhost:15672/#/](http://localhost:15672) and the flower dashboard running at [http://localhost:5556](http://localhost:5556)

### Trigger a task

Crawl company data

```sh
curl http://localhost:8001/companies
```

Analyze company data
```sh
curl http://localhost:8001/analyze
```

