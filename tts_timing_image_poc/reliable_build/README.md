```
colima start --cpu 4 --memory 8 --disk 100

docker build -f Dockerfile.cpu -t model-runner-cpu .

docker build -f Dockerfile.gpu -t model-runner-gpu .
```
