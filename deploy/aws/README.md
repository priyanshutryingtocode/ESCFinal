# AWS ECS demo deployment

This deploys the Gradio audio upload demo as an Amazon ECS Express Mode service.
The Docker image includes the model checkpoints from `models/`, so build it
from this project folder on a machine that has those files.

## Prerequisites

- AWS CLI v2 configured for an AWS account with permission to use ECR and ECS.
- Docker Desktop running.
- An ECS task execution role and an ECS Express Mode infrastructure role.
  The infrastructure role needs permission to provision and manage the load
  balancer, networking, and autoscaling resources for the service.
- An AWS region that supports ECS Express Mode and Fargate.

## Build and publish the image (PowerShell)

The demo is deployed in `ap-southeast-2`. The ECR repository and immutable
`demo-v2` image have been published there. To publish another image, use a new
tag because the repository rejects tag overwrites.

```powershell
$Region = "ap-southeast-2"
$Repository = "esc50-demo"
$ImageTag = "demo-v2"
$AccountId = aws sts get-caller-identity --query Account --output text
$Image = "${AccountId}.dkr.ecr.${Region}.amazonaws.com/${Repository}:${ImageTag}"

aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin "${AccountId}.dkr.ecr.${Region}.amazonaws.com"

docker build -t "$($Repository):latest" .
docker tag "$($Repository):latest" $Image
docker push $Image
```

The container includes the model checkpoints; keep the model files available
in the build context.

## ECS Express Mode service

The task execution and infrastructure roles are already created in account
`164892691020`. Their trust policies are in
`ecs-task-execution-trust.json` and `ecs-express-infrastructure-trust.json`.
The service definition in `ecs-express-service.json` contains the current
deployed image and role ARNs. It configures 1 vCPU, 4 GiB memory, and one to
two tasks.

To recreate the service after deleting it, run from the repository root:

```powershell
aws ecs create-express-gateway-service `
  --cli-input-json file://deploy/aws/ecs-express-service.json `
  --monitor-resources `
  --region ap-southeast-2
```

The live demo is available at:
https://es-d3438eb0c9d640c38cbe34d106f62a87.ecs.ap-southeast-2.on.aws

The `/health` endpoint returns `{"status":"ok"}`.

## Cost and cleanup

This demo currently keeps one Fargate task running (1 vCPU, 4 GiB) and can scale
to two. The task and its Application Load Balancer can incur charges while the
demo is idle. Setting `maxTaskCount` to `1` limits possible scale-out cost, but
does not reduce the always-on baseline because the minimum remains one.

For an occasional demo, the largest idle-time saving is to delete the Express
service when it is not needed and recreate it before presenting. Deleting an
Express service removes its tasks and its associated load balancer and managed
networking resources; the ECR image remains available for the next deployment.
Recreation takes time and the demo URL may change. Check the AWS Pricing
Calculator for the selected region before creating the service.

To stop those service charges, delete the Express Mode service using its ARN:

```powershell
aws ecs delete-express-gateway-service --service-arn <SERVICE_ARN> --region $Region
```

The container image remains in ECR and may incur a small storage charge. Delete
the image or ECR repository separately if it is no longer needed.
