# Deploying to Amazon SageMaker (serverless)

This runs the released CPU image as an Amazon SageMaker **Serverless Inference** endpoint: AWS starts the container
when a request arrives and bills only while it's handling requests. The endpoint is meant to be deployed, tested and
deleted again, not left running. [`scripts/sagemaker.py`](../scripts/sagemaker.py) does all of it.

## What it costs

- **The endpoint:** no charge while idle. You pay per millisecond of processing, scaled by the memory size
  (3 GB by default). A prediction takes a fraction of a second of CPU, so a test session of a few dozen calls costs
  cents. New AWS accounts have usually had a free allowance for serverless inference; check the
  [SageMaker pricing page](https://aws.amazon.com/sagemaker/pricing/) for current numbers.
- **The image in ECR:** about $0.10 per GB per month, so about 2 cents a month for the ~200 MB image.
- **Logs in CloudWatch:** negligible for a test.

`delete --everything` removes all three, so nothing keeps costing money afterwards.

## Limits worth knowing

- **CPU only.** Serverless endpoints have no GPUs, so this uses the CPU image.
- **Cold starts.** The first request after a quiet spell takes several seconds while AWS starts the container;
  calls right after it are fast. `invoke --repeat 3` shows the difference.
- **Requests up to 4 MB** and **60 seconds**. MRI slices are tens of kilobytes, far below both.

## One-off setup (in your AWS account)

These steps involve your account, payment details and keys, so they're yours to do.

1. **Set a budget alert first.** In the AWS console, go to **Billing and Cost Management → Budgets → Create budget**
   and pick the **Monthly cost budget** template with, for example, $5. AWS then emails you if spending ever
   approaches it.
2. **Pick a region** close to you, for example `ap-southeast-2` (Sydney) or `us-east-1` (N. Virginia), and use it
   for everything below.
3. **Install the AWS CLI and log in.** Install it (`winget install Amazon.AWSCLI` on Windows), then run
   `aws configure` in your own terminal and enter an access key of an IAM user, never of the root account.
   `aws sts get-caller-identity` should then print your account number.
4. **Create the SageMaker execution role**, the identity the endpoint runs as. In the console: **IAM → Roles →
   Create role → AWS service → SageMaker**, keep the suggested `AmazonSageMakerFullAccess` policy, name it
   `btd-sagemaker-execution`, and create it. Copy its **ARN** (`arn:aws:iam::<account>:role/btd-sagemaker-execution`).
5. **Install boto3** in your environment and make sure **Docker Desktop is running**:

   ```bash
   pip install -e ".[aws]"
   ```

## Deploy, test, delete

Use a released tag that includes the SageMaker routes (0.1.3 or later), without the `v`:

```bash
python scripts/sagemaker.py push --tag 0.1.3
python scripts/sagemaker.py deploy --tag 0.1.3 --role-arn arn:aws:iam::<account>:role/btd-sagemaker-execution
python scripts/sagemaker.py invoke --repeat 3
python scripts/sagemaker.py delete --everything
```

Add `--region <region>` to every command if it differs from your `aws configure` default.

- **`push`** creates a private ECR repository (with vulnerability scanning on) and copies the linux/amd64 image
  from GHCR into it. SageMaker only runs images from ECR in your own account and region. It also rejects both the
  multi-part index GHCR serves (image plus SBOM and provenance) and the newer OCI image format
  (`Unsupported manifest media type application/vnd.oci.image.manifest.v1+json`), so `push` re-packs the image in
  Docker's own v2 format. The layers, entrypoint, user and environment stay identical; only the packaging changes,
  and `push` checks the result before you deploy.
- **`deploy`** creates a SageMaker model from that image, a serverless endpoint configuration (3 GB, one request at
  a time; change with `--memory` and `--max-concurrency`) and the endpoint, then waits until it's InService. That
  usually takes a few minutes.
- **`invoke`** sends the glioma sample slice (or `--file`) and prints the decision, any input warnings, the model's
  own processing time and the full round trip.
- **`status`** shows whether the endpoint exists and its state.
- **`delete`** removes the endpoint, its configuration and the model. `--everything` also removes the ECR repository
  and the endpoint's CloudWatch log group.

## How the container fits SageMaker

| SageMaker expects | What this image does |
|---|---|
| It starts the container as `docker run IMAGE serve` | The image's entrypoint is `btd`, so that runs `btd serve` |
| The server listens on port 8080 | `deploy` sets `BTD_PORT=8080` (8000 is the default elsewhere) |
| `GET /ping` answers 200 when ready | Added with `BTD_SAGEMAKER=true`; 503 until the model is loaded |
| `POST /invocations` with the raw request body | The same response as `/v1/predict`, including `warnings` |
| The model comes with the container or from S3 | It's built into the image, so no S3 bucket is needed |

`/ping` and `/invocations` only exist when `BTD_SAGEMAKER=true`. On SageMaker, AWS checks every caller's
credentials before a request reaches the container. Anywhere else, `/invocations` would be a second way in that
bypasses `BTD_API_KEY`, so ordinary deployments don't get it. Don't set `BTD_API_KEY` on SageMaker: its client can't
send the header.

CI tests this contract on every push: it starts the image as `docker run IMAGE serve` with `BTD_SAGEMAKER=true` and
`BTD_PORT=8080` and runs [`scripts/smoke_test.py --sagemaker`](../scripts/smoke_test.py) against it.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `No AWS region set` | Run `aws configure` or add `--region` |
| `push` fails at `docker login` or with "denied" | Check Docker Desktop is running and `aws sts get-caller-identity` works |
| `deploy` fails with an ECR access error | Attach the `AmazonEC2ContainerRegistryReadOnly` policy to the execution role |
| The endpoint ends up `Failed` | `deploy` prints the reason; the container's own logs are in CloudWatch under `/aws/sagemaker/Endpoints/brain-tumor-detection` |
| `invoke` returns `ModelError` | Look at the same CloudWatch log group; the API logs each request as JSON |
| `already exists` on `deploy` | An earlier endpoint is still there; run `delete` first |
