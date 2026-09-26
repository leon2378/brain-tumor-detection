# Deploying to Amazon SageMaker

This runs the released CPU image on an Amazon SageMaker endpoint, meant to be deployed, tested and deleted again,
not left running. [`scripts/sagemaker.py`](../scripts/sagemaker.py) does all of it. Two kinds of endpoint are
supported: **real-time**, on a dedicated instance billed while it exists, and **serverless**, which starts the
container per request and bills only for processing time.

## Status (September 2026)

**Real-time works.** Deployed from the released `0.1.4` image in `ap-southeast-2` on one `ml.t2.medium`:

| | Result |
|---|---|
| Time to InService | under 2 minutes |
| Container start-up | model loaded on the CPU, no errors; SageMaker's `/ping` health checks all answered |
| `invoke --repeat 3` with the glioma sample slice | 3 × HTTP 200, 359–385 ms each inside the container |

`ml.t2.medium` is a small, burstable CPU, roughly five times slower than a Ryzen 7 5800H laptop for this model.

**Serverless doesn't work yet.** Both attempts ended `Failed` after about 6.5 minutes with only "Request to service
failed. If failure persists after retry, contact customer support.", and no container logs were ever written. Ruled
out so far:

- **Image format:** the image in ECR is a single Docker v2 manifest, which SageMaker accepted when creating the model.
- **Start-up in a Lambda-style sandbox:** the image starts with a replaced `PATH`, an arbitrary user and a read-only
  filesystem (this fixed a real bug in 0.1.3, but not the serverless failure).
- **Permissions:** the execution role has `AmazonSageMakerFullAccess`, trusts `sagemaker.amazonaws.com`, and the IAM
  policy simulator allows it to pull the image and write logs.
- **Account limits:** serverless concurrency 10 and 25 endpoints, far above the 1 and 1 requested.

The same image, role and settings run on a real-time endpoint, so the cause is specific to serverless. The next step
is reading the serverless attempt in CloudTrail (needs `AWSCloudTrail_ReadOnlyAccess`) to see which step failed.

## What it costs

- **A real-time endpoint:** billed per second while it exists, whether or not it's used. `ml.t2.medium` is roughly
  5–10 US cents an hour, so a 15-minute test costs a few cents. Delete it as soon as you're done.
- **A serverless endpoint:** no charge while idle. You pay per millisecond of processing, scaled by the memory size
  (3 GB by default). A prediction takes a fraction of a second of CPU, so a test session of a few dozen calls costs
  cents. New AWS accounts have usually had a free allowance for serverless inference.
- Check the [SageMaker pricing page](https://aws.amazon.com/sagemaker/pricing/) for current numbers.
- **The image in ECR:** about $0.10 per GB per month, so about 2 cents a month for the ~200 MB image.
- **Logs in CloudWatch:** negligible for a test.

`delete --everything` removes all three, so nothing keeps costing money afterwards.

## Limits of serverless

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

Use a released tag that includes the SageMaker routes and the full-path entrypoint (0.1.4 or later), without the `v`.
The working route today is a real-time endpoint:

```bash
python scripts/sagemaker.py push --tag 0.1.4
python scripts/sagemaker.py deploy --tag 0.1.4 --role-arn arn:aws:iam::<account>:role/btd-sagemaker-execution --instance-type ml.t2.medium
python scripts/sagemaker.py invoke --repeat 3
python scripts/sagemaker.py delete --everything
```

Leave out `--instance-type` for a serverless endpoint (see the status above). Add `--region <region>` to every
command if it differs from your `aws configure` default. To send your own slice, use
`invoke --file path/to/slice.jpg`, or call it directly:

```bash
aws sagemaker-runtime invoke-endpoint --endpoint-name brain-tumor-detection --content-type application/x-image --body fileb://slice.jpg result.json
```

The web page can't be used with a SageMaker endpoint: every request must be signed with AWS credentials, so the
image runs there with `BTD_UI=false`.

- **`push`** creates a private ECR repository (with vulnerability scanning on) and copies the linux/amd64 image
  from GHCR into it. SageMaker only runs images from ECR in your own account and region. It also rejects both the
  multi-part index GHCR serves (image plus SBOM and provenance) and the newer OCI image format
  (`Unsupported manifest media type application/vnd.oci.image.manifest.v1+json`), so `push` re-packs the image in
  Docker's own v2 format. The layers, entrypoint, user and environment stay identical; only the packaging changes,
  and `push` checks the result before you deploy.
- **`deploy`** creates a SageMaker model from that image, an endpoint configuration and the endpoint, then waits
  until it's InService. With `--instance-type` it's a real-time endpoint on one instance of that type (billed while
  it exists); without it, a serverless one (3 GB, one request at a time; change with `--memory` and
  `--max-concurrency`).
- **`invoke`** sends the glioma sample slice (or `--file`) and prints the decision, any input warnings, the model's
  own processing time and the full round trip.
- **`status`** shows whether the endpoint exists and its state.
- **`delete`** removes the endpoint, its configuration and the model. `--everything` also removes the ECR repository
  and the endpoint's CloudWatch log group.

## How the container fits SageMaker

| SageMaker expects | What this image does |
|---|---|
| It starts the container as `docker run IMAGE serve` | The entrypoint is `/opt/venv/bin/btd`, so that runs `btd serve` |
| It replaces the image's `PATH` and runs as its own user on a read-only filesystem | The entrypoint is a full path, and the server writes nothing outside `/tmp` |
| The server listens on port 8080 | `deploy` sets `BTD_PORT=8080` (8000 is the default elsewhere) |
| `GET /ping` answers 200 when ready | Added with `BTD_SAGEMAKER=true`; 503 until the model is loaded |
| `POST /invocations` with the raw request body | The same response as `/v1/predict`, including `warnings` |
| The model comes with the container or from S3 | It's built into the image, so no S3 bucket is needed |

`/ping` and `/invocations` only exist when `BTD_SAGEMAKER=true`. On SageMaker, AWS checks every caller's
credentials before a request reaches the container. Anywhere else, `/invocations` would be a second way in that
bypasses `BTD_API_KEY`, so ordinary deployments don't get it. Don't set `BTD_API_KEY` on SageMaker: its client can't
send the header.

CI tests this contract on every push: it starts the image as `docker run IMAGE serve` with `BTD_SAGEMAKER=true` and
`BTD_PORT=8080`, with Lambda's `PATH`, an arbitrary user and a read-only filesystem, and runs
[`scripts/smoke_test.py --sagemaker`](../scripts/smoke_test.py) against it.

That sandbox check exists because of a real failure. In 0.1.3 the entrypoint was a bare `btd`, found only through the
image's `PATH`. SageMaker Serverless replaces `PATH`, so the container couldn't start, wrote no logs, and the endpoint
failed after about six minutes with only "Request to service failed". 0.1.4 uses the full path.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `No AWS region set` | Run `aws configure` or add `--region` |
| `push` fails at `docker login` or with "denied" | Check Docker Desktop is running and `aws sts get-caller-identity` works |
| `deploy` fails with an ECR access error | Attach the `AmazonEC2ContainerRegistryReadOnly` policy to the execution role |
| The endpoint ends up `Failed` | `deploy` prints the reason; the container's own logs are in CloudWatch under `/aws/sagemaker/Endpoints/brain-tumor-detection` |
| `Failed` with only "Request to service failed", and no log group exists | The container never started. Check the image starts with a replaced `PATH`: `docker run --rm -e PATH=/usr/local/bin:/usr/bin/:/bin:/opt/bin -e BTD_SAGEMAKER=true -e BTD_PORT=8080 IMAGE serve`. If it does, deploy with `--instance-type ml.t2.medium`: a real-time endpoint gives detailed errors and logs. If that works too, the problem is specific to serverless (see the status at the top) |
| `invoke` returns `ModelError` | Look at the same CloudWatch log group; the API logs each request as JSON |
| `already exists` on `deploy` | An earlier endpoint is still there; run `delete` first |
