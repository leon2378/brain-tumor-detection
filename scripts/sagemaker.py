"""Deploy the CPU image to an Amazon SageMaker Serverless Inference endpoint, try it, and delete it again.

    pip install -e ".[aws]"                            # boto3
    aws configure                                      # your own credentials; this script never sees your keys
    python scripts/sagemaker.py push --tag 0.1.3       # GHCR image -> a private ECR repository
    python scripts/sagemaker.py deploy --tag 0.1.3 --role-arn arn:aws:iam::<account>:role/<execution role>
    python scripts/sagemaker.py invoke [--file slice.jpg] [--repeat 3]
    python scripts/sagemaker.py status
    python scripts/sagemaker.py delete [--everything]

push    copies the linux/amd64 image of ghcr.io/leon2378/brain-tumor-detection:<tag> into ECR, because SageMaker only
        runs images from ECR in your own account and region. SageMaker also rejects both the multi-part index GHCR
        serves (image plus SBOM and provenance) and OCI image manifests, so the image is re-packed as a plain Docker
        v2 image: same layers, entrypoint, user and environment, only the packaging format differs.
deploy  creates the model, a serverless endpoint configuration and the endpoint, then waits until it's InService.
        The container runs as `docker run IMAGE serve` with BTD_SAGEMAKER=true and BTD_PORT=8080.
invoke  sends an MRI slice (default: the glioma sample) and prints the decision and the timings.
delete  removes the endpoint, its configuration and the model. --everything also removes the ECR repository and
        the endpoint's CloudWatch log group, leaving nothing behind.

The region is --region or your AWS CLI default. See docs/SAGEMAKER.md for the one-off AWS setup and the costs.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

REPO = Path(__file__).resolve().parents[1]
SOURCE_IMAGE = "ghcr.io/leon2378/brain-tumor-detection"
DEFAULT_SLICE = REPO / "src" / "btd" / "api" / "static" / "samples" / "brisc2025_test_00042_gl_ax_t1.jpg"
CONTAINER_ENV = {
    "BTD_SAGEMAKER": "true",  # adds /ping and /invocations
    "BTD_PORT": "8080",  # the port SageMaker talks to
    "BTD_UI": "false",  # nobody browses to an endpoint
    "BTD_LOG_JSON": "true",
}


def session_for(args: argparse.Namespace) -> tuple[boto3.Session, str, str]:
    session = boto3.Session(region_name=args.region)
    if not session.region_name:
        sys.exit("No AWS region set: pass --region or run `aws configure`")
    account = session.client("sts").get_caller_identity()["Account"]
    return session, session.region_name, account


def registry_of(account: str, region: str) -> str:
    return f"{account}.dkr.ecr.{region}.amazonaws.com"


def run(cmd: list[str], **kwargs: Any) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs).stdout


def amd64_digest(image: str) -> str:
    """Digest of the linux/amd64 image manifest inside a (possibly multi-platform) image."""
    raw = json.loads(run(["docker", "buildx", "imagetools", "inspect", "--raw", image]))
    if "manifests" not in raw:
        return run(
            ["docker", "buildx", "imagetools", "inspect", image, "--format", "{{.Manifest.Digest}}"]
        ).strip()
    for entry in raw["manifests"]:
        platform = entry.get("platform", {})
        if platform.get("os") == "linux" and platform.get("architecture") == "amd64":
            return str(entry["digest"])
    sys.exit(f"{image} has no linux/amd64 image")


def cmd_push(args: argparse.Namespace) -> None:
    session, region, account = session_for(args)
    ecr = session.client("ecr")
    try:
        ecr.create_repository(repositoryName=args.repository, imageScanningConfiguration={"scanOnPush": True})
        print(f"created ECR repository {args.repository}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        pass

    # A short-lived ECR login token (not your AWS keys), handed to `docker login` on stdin.
    token = ecr.get_authorization_token()["authorizationData"][0]["authorizationToken"]
    password = base64.b64decode(token).decode().split(":", 1)[1]
    registry = registry_of(account, region)
    run(["docker", "login", "--username", "AWS", "--password-stdin", registry], input=password)

    source = f"{SOURCE_IMAGE}:{args.tag}"
    digest = amd64_digest(source)
    target = f"{registry}/{args.repository}:{args.tag}"
    print(f"copying {source} (linux/amd64, {digest[:19]}...) to {target}")
    # A build that is only `FROM` the published image keeps its layers and config unchanged; exporting it with
    # oci-mediatypes=false and no attestations yields the Docker v2 manifest SageMaker accepts.
    run(
        [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64",
            "--provenance=false",
            "--sbom=false",
            "--output",
            f"type=image,name={target},push=true,oci-mediatypes=false",
            "-",
        ],
        input=f"FROM {SOURCE_IMAGE}@{digest}\n",
    )
    check_manifest(ecr, args.repository, args.tag)
    print(f"pushed {target}")


def check_manifest(ecr: Any, repository: str, tag: str) -> None:
    """Fail early, before `deploy`, if the pushed image isn't in a format SageMaker accepts."""
    image = ecr.batch_get_image(repositoryName=repository, imageIds=[{"imageTag": tag}])["images"][0]
    media_type = image.get("imageManifestMediaType") or json.loads(image["imageManifest"]).get("mediaType")
    if media_type != "application/vnd.docker.distribution.manifest.v2+json":
        sys.exit(f"pushed image has manifest type {media_type}, which SageMaker rejects")


def cmd_deploy(args: argparse.Namespace) -> None:
    session, region, account = session_for(args)
    sm = session.client("sagemaker")
    image = args.image_uri or f"{registry_of(account, region)}/{args.repository}:{args.tag}"
    print(f"creating model {args.name} from {image}")
    try:
        sm.create_model(
            ModelName=args.name,
            ExecutionRoleArn=args.role_arn,
            PrimaryContainer={"Image": image, "Environment": CONTAINER_ENV},
        )
        sm.create_endpoint_config(
            EndpointConfigName=args.name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": args.name,
                    "ServerlessConfig": {
                        "MemorySizeInMB": args.memory,
                        "MaxConcurrency": args.max_concurrency,
                    },
                }
            ],
        )
        sm.create_endpoint(EndpointName=args.name, EndpointConfigName=args.name)
    except ClientError as err:
        if "already exist" in str(err):
            sys.exit(f"{args.name} already exists - run `delete` first, or pass --name")
        raise

    print(
        f"waiting for endpoint {args.name} ({args.memory} MB, concurrency {args.max_concurrency}); usually 3-8 minutes"
    )
    started = time.time()
    while True:
        desc = sm.describe_endpoint(EndpointName=args.name)
        state = desc["EndpointStatus"]
        if state == "InService":
            print(f"InService after {time.time() - started:.0f} s - try: python scripts/sagemaker.py invoke")
            return
        if state == "Failed":
            sys.exit(f"endpoint failed: {desc.get('FailureReason', 'no reason given')}")
        time.sleep(20)


def cmd_invoke(args: argparse.Namespace) -> None:
    session, _, _ = session_for(args)
    runtime = session.client("sagemaker-runtime")
    data = Path(args.file).read_bytes()
    for i in range(args.repeat):
        start = time.perf_counter()
        resp = runtime.invoke_endpoint(
            EndpointName=args.name, ContentType="application/x-image", Accept="application/json", Body=data
        )
        elapsed = (time.perf_counter() - start) * 1000
        body = json.loads(resp["Body"].read())
        d = body["decision"]
        warnings = ",".join(w["code"] for w in body.get("warnings", [])) or "none"
        print(
            f"call {i + 1}: {d['label']} {d['score']:.2f} | {len(body['detections'])} detection(s) | "
            f"warnings: {warnings} | model {body['timings_ms'].get('total_ms', 0):.0f} ms, "
            f"round trip {elapsed:.0f} ms"
        )


def cmd_status(args: argparse.Namespace) -> None:
    session, region, _ = session_for(args)
    sm = session.client("sagemaker")
    try:
        desc = sm.describe_endpoint(EndpointName=args.name)
    except ClientError:
        print(f"no endpoint named {args.name} in {region}")
        return
    print(
        f"{args.name} in {region}: {desc['EndpointStatus']} (created {desc['CreationTime']:%Y-%m-%d %H:%M})"
    )


def cmd_delete(args: argparse.Namespace) -> None:
    session, region, _ = session_for(args)
    sm = session.client("sagemaker")
    steps = [
        ("endpoint", lambda: sm.delete_endpoint(EndpointName=args.name)),
        ("endpoint configuration", lambda: sm.delete_endpoint_config(EndpointConfigName=args.name)),
        ("model", lambda: sm.delete_model(ModelName=args.name)),
    ]
    if args.everything:
        ecr, logs = session.client("ecr"), session.client("logs")
        steps += [
            ("ECR repository", lambda: ecr.delete_repository(repositoryName=args.repository, force=True)),
            (
                "log group",
                lambda: logs.delete_log_group(logGroupName=f"/aws/sagemaker/Endpoints/{args.name}"),
            ),
        ]
    for what, delete in steps:
        try:
            delete()
            print(f"deleted {what}")
        except ClientError as err:
            print(f"no {what} to delete ({err.response['Error']['Code']})")
    print(f"done ({region})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", help="default: your AWS CLI region")
    ap.add_argument("--name", default="brain-tumor-detection", help="model, config and endpoint name")
    ap.add_argument("--repository", default="brain-tumor-detection", help="ECR repository")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("push", help="copy the GHCR image into ECR")
    p.add_argument("--tag", required=True, help="release tag without the v, e.g. 0.1.3")
    p.set_defaults(func=cmd_push)

    d = sub.add_parser("deploy", help="create the serverless endpoint")
    d.add_argument("--role-arn", required=True, help="SageMaker execution role")
    d.add_argument("--tag", help="ECR tag pushed with `push`")
    d.add_argument("--image-uri", help="full image URI instead of --tag")
    d.add_argument("--memory", type=int, default=3072, choices=[1024, 2048, 3072, 4096, 5120, 6144])
    d.add_argument("--max-concurrency", type=int, default=1)
    d.set_defaults(func=cmd_deploy)

    i = sub.add_parser("invoke", help="send an MRI slice")
    i.add_argument("--file", default=str(DEFAULT_SLICE))
    i.add_argument(
        "--repeat", type=int, default=1, help="more than 1 shows the cold start against warm calls"
    )
    i.set_defaults(func=cmd_invoke)

    sub.add_parser("status", help="show the endpoint's state").set_defaults(func=cmd_status)

    r = sub.add_parser("delete", help="remove the endpoint")
    r.add_argument("--everything", action="store_true", help="also the ECR repository and the log group")
    r.set_defaults(func=cmd_delete)

    args = ap.parse_args()
    if args.command == "deploy" and not (args.tag or args.image_uri):
        ap.error("deploy needs --tag or --image-uri")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
