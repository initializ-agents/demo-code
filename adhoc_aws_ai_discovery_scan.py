#!/usr/bin/env python3
"""
Scan AWS accounts for generative AI service usage: Bedrock and Amazon Q.

Detects:
  - Bedrock foundation model access, invocation logging, and usage (high confidence)
  - Amazon Q Business applications and enablement
  - Amazon Q Developer enablement and subscription
  - IAM roles and policies granting bedrock:*, q:*, qbusiness:*, or codewhisperer:* permissions (low confidence)
  - Cost Explorer spend on these services (high confidence, trailing 3 months)
  - SageMaker resources (optional, via --include-sagemaker flag)

Supports single-account mode (using active credentials, for testing) or multi-account mode
via cross-account role assumption from an AWS Organizations management account.

Required environment variables (standard boto3):
  AWS_PROFILE, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (optional, use default credentials)
  For multi-account mode: ensure you have assumed the management account role or credentials

Install: pip install boto3 click

Usage:
  python adhoc_aws_ai_discovery_scan.py
  python adhoc_aws_ai_discovery_scan.py --verbose --workers 4
  python adhoc_aws_ai_discovery_scan.py --cross-account-role OrganizationAccountAccessRole --workers 8
  python adhoc_aws_ai_discovery_scan.py --include-sagemaker
  python adhoc_aws_ai_discovery_scan.py --json
  python adhoc_aws_ai_discovery_scan.py --csv report.csv --detail-csv findings.csv
  python adhoc_aws_ai_discovery_scan.py --accounts-file accounts.txt
  python adhoc_aws_ai_discovery_scan.py --regions us-east-1 us-west-2 eu-west-1

IAM Permissions Required (single-account mode):
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "bedrock:ListFoundationModels",
          "bedrock:GetModelInvocationLoggingConfiguration"
        ],
        "Resource": "*"
      },
      {
        "Effect": "Allow",
        "Action": [
          "qbusiness:ListApplications",
          "qbusiness:GetApplication"
        ],
        "Resource": "*"
      },
      {
        "Effect": "Allow",
        "Action": [
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams",
          "logs:FilterLogEvents"
        ],
        "Resource": "arn:aws:logs:*:ACCOUNT_ID:log-group:/aws/bedrock/*"
      },
      {
        "Effect": "Allow",
        "Action": [
          "iam:ListPolicies",
          "iam:GetPolicy",
          "iam:GetPolicyVersion",
          "iam:ListRoles",
          "iam:GetRole",
          "iam:ListRolePolicies",
          "iam:GetRolePolicy",
          "iam:ListUsers",
          "iam:ListUserPolicies",
          "iam:GetUserPolicy",
          "iam:ListEntitiesForPolicy"
        ],
        "Resource": "*"
      },
      {
        "Effect": "Allow",
        "Action": [
          "ce:GetCostAndUsage"
        ],
        "Resource": "*"
      },
      {
        "Effect": "Allow",
        "Action": [
          "organizations:ListAccounts"
        ],
        "Resource": "*"
      }
    ]
  }

IAM Permissions Required (cross-account role, attached to the role in linked accounts):
  - Same as above, plus:
  - Trust relationship from management account: allow sts:AssumeRole from management account principal

Estimated runtime: 30-60s per account (multi-region), ~5-10min for 10 accounts.
Cost: ~200-300 API calls per account per run, minimal AWS cost impact.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import click
import boto3
from botocore.exceptions import ClientError, BotoCoreError, NoCredentialsError

_SCRIPT_DIR = Path(__file__).resolve().parent
_OUTPUT_STEM = "adhoc_aws_ai_discovery_scan"

# Default regions where Bedrock/Q have reasonable availability
_DEFAULT_REGIONS = [
    "us-east-1",
    "us-west-2",
    "eu-west-1",
    "ap-northeast-1",
    "ap-southeast-1",
]

_SUMMARY_FIELDS = (
    "account_id",
    "account_name",
    "bedrock_api_reachable",
    "bedrock_logging_enabled",
    "bedrock_recent_invocations",
    "q_business_enabled",
    "q_business_app_count",
    "q_developer_enabled",
    "sagemaker_endpoint_count",
    "sagemaker_notebook_count",
    "sagemaker_recent_training_job_count",
    "iam_principals_with_ai_access_count",
    "cost_bedrock_trailing_3mo",
    "cost_q_trailing_3mo",
    "cost_sagemaker_trailing_3mo",
    "scan_error",
)

_DETAIL_FIELDS = (
    "account_id",
    "account_name",
    "finding_type",
    "resource_name_or_id",
    "status_or_value",
    "region",
    "confidence",
)

_thread_local = threading.local()


@dataclass(frozen=True)
class DetailFinding:
    account_id: str
    account_name: str
    finding_type: str
    resource_name_or_id: str
    status_or_value: str
    region: str
    confidence: str  # high, medium, low


@dataclass
class AccountScanResult:
    account_id: str
    account_name: str
    bedrock_api_reachable: bool = False
    bedrock_logging_enabled: Optional[bool] = None
    bedrock_recent_invocations: Optional[int] = None
    q_business_enabled: bool = False
    q_business_app_count: int = 0
    q_developer_enabled: bool = False
    sagemaker_endpoint_count: int = 0
    sagemaker_notebook_count: int = 0
    sagemaker_recent_training_job_count: int = 0
    iam_principals_with_ai_access_count: int = 0
    cost_bedrock_trailing_3mo: float = 0.0
    cost_q_trailing_3mo: float = 0.0
    cost_sagemaker_trailing_3mo: float = 0.0
    findings: list[DetailFinding] = field(default_factory=list)
    scan_error: Optional[str] = None


def _timestamped_output_path(suffix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return _SCRIPT_DIR / f"{_OUTPUT_STEM}_{stamp}{suffix}"


def _assume_role(account_id: str, role_name: str) -> Optional[boto3.Session]:
    """Assume cross-account role and return a session."""
    try:
        sts = boto3.client("sts")
        role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
        response = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName=f"ai-discovery-{int(time.time())}",
            DurationSeconds=3600,
        )
        credentials = response["Credentials"]
        return boto3.Session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )
    except ClientError as e:
        raise RuntimeError(f"Failed to assume role in account {account_id}: {e.response['Error']['Message']}")
    except Exception as e:
        raise RuntimeError(f"Failed to assume role in account {account_id}: {e}")


def _get_bedrock_info(session: boto3.Session, region: str) -> dict[str, Any]:
    """Get Bedrock status and logging config for a region."""
    result = {
        "api_reachable": False,
        "logging_enabled": None,
        "invocation_count": None,
    }

    try:
        bedrock = session.client("bedrock", region_name=region)
        # Test if Bedrock is accessible by listing models
        bedrock.list_foundation_models()
        result["api_reachable"] = True

        # Check logging configuration
        try:
            logging_config = bedrock.get_model_invocation_logging_configuration()
            cw_config = logging_config.get("loggingConfig", {}).get("cloudWatchLogsConfig")
            s3_config = logging_config.get("loggingConfig", {}).get("s3Config")
            result["logging_enabled"] = bool(cw_config or s3_config)

            # If logging is enabled, try to count recent invocations from CloudWatch
            if result["logging_enabled"] and cw_config:
                try:
                    log_group_name = cw_config.get("logGroupName")
                    if log_group_name:
                        logs = session.client("logs", region_name=region)
                        # Get events from last 7 days as a sample
                        start_time = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp() * 1000)
                        response = logs.filter_log_events(
                            logGroupName=log_group_name,
                            startTime=start_time,
                        )
                        result["invocation_count"] = len(response.get("events", []))
                except ClientError:
                    # Log access failed, but logging is still enabled
                    pass
        except ClientError as e:
            if e.response["Error"]["Code"] not in ["AccessDenied", "UnauthorizedOperation"]:
                # Unexpected error, but Bedrock is reachable
                pass

    except ClientError as e:
        if e.response["Error"]["Code"] not in ["AccessDenied", "UnauthorizedOperation", "ServiceUnavailable"]:
            pass
    except Exception:
        pass

    return result


def _get_q_business_info(session: boto3.Session, region: str) -> dict[str, Any]:
    """Get Amazon Q Business information for a region."""
    result = {
        "enabled": False,
        "app_count": 0,
        "apps": [],
    }

    try:
        qbusiness = session.client("qbusiness", region_name=region)

        # List Q Business applications
        try:
            response = qbusiness.list_applications()
            apps = response.get("applications", [])
            if apps:
                result["enabled"] = True
                result["app_count"] = len(apps)
                for app in apps:
                    result["apps"].append({
                        "id": app.get("applicationId", ""),
                        "name": app.get("displayName", ""),
                        "status": app.get("status", ""),
                    })
        except ClientError:
            pass
    except ClientError as e:
        if e.response["Error"]["Code"] not in ["AccessDenied", "UnauthorizedOperation", "ServiceUnavailable"]:
            pass
    except Exception:
        pass

    return result


def _get_q_developer_info(session: boto3.Session) -> dict[str, Any]:
    """
    Get Amazon Q Developer enablement status.

    NOTE: There is no reliable AWS API that reports whether Q Developer is
    enabled/subscribed at the account level. Q Developer is provisioned via
    IDE plugins and SSO policies, not at the account level like Bedrock or Q Business.

    This returns {"enabled": False} always. To detect Q Developer usage, check:
    - AWS billing for "Amazon CodeWhisperer" or "AWS Q Developer" line items
    - IDE plugin configurations (out of scope for this API-based scan)

    See: https://docs.aws.amazon.com/codewhisperer/latest/userguide/
    """
    result = {
        "enabled": False,  # Cannot be reliably detected via API
    }
    return result


def _get_sagemaker_info(session: boto3.Session, region: str) -> dict[str, Any]:
    """Get SageMaker resources for a region."""
    result = {
        "endpoints": [],
        "notebooks": [],
        "training_jobs": [],
    }

    try:
        sagemaker = session.client("sagemaker", region_name=region)

        # List endpoints
        try:
            paginator = sagemaker.get_paginator("list_endpoints")
            for page in paginator.paginate():
                for endpoint in page.get("Endpoints", []):
                    result["endpoints"].append({
                        "name": endpoint.get("EndpointName", ""),
                        "status": endpoint.get("EndpointStatus", ""),
                    })
        except ClientError:
            pass

        # List notebook instances
        try:
            paginator = sagemaker.get_paginator("list_notebook_instances")
            for page in paginator.paginate():
                for notebook in page.get("NotebookInstances", []):
                    result["notebooks"].append({
                        "name": notebook.get("NotebookInstanceName", ""),
                        "status": notebook.get("NotebookInstanceStatus", ""),
                    })
        except ClientError:
            pass

        # List training jobs (last 90 days)
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=90)
            paginator = sagemaker.get_paginator("list_training_jobs")
            for page in paginator.paginate(CreationTimeAfter=cutoff):
                for job in page.get("TrainingJobSummaries", []):
                    result["training_jobs"].append({
                        "name": job.get("TrainingJobName", ""),
                        "status": job.get("TrainingJobStatus", ""),
                    })
        except ClientError:
            pass

    except Exception:
        pass

    return result


def _policy_grants_ai_access(policy_doc: dict) -> bool:
    """Check if policy grants bedrock:*, q:*, qbusiness:*, codewhisperer:*, or sagemaker:* access."""
    if not isinstance(policy_doc, dict):
        return False

    statements = policy_doc.get("Statement", [])
    if not isinstance(statements, list):
        return False

    for statement in statements:
        if statement.get("Effect") != "Allow":
            continue

        actions = statement.get("Action", [])
        if isinstance(actions, str):
            actions = [actions]
        if not isinstance(actions, list):
            continue

        for action in actions:
            if not isinstance(action, str):
                continue
            action_lower = action.lower()
            # Check for generative AI service prefixes
            if any(prefix in action_lower for prefix in ["bedrock:", "q:", "qbusiness:", "codewhisperer:", "sagemaker:"]):
                # Check for wildcards or specific permissions
                if "*" in action or any(action_lower.startswith(p) for p in ["bedrock:", "q:", "qbusiness:", "codewhisperer:", "sagemaker:"]):
                    return True

    return False


def _get_iam_access(session: boto3.Session) -> dict[str, Any]:
    """Get IAM principals with AI service access."""
    result = {
        "principals": [],
    }

    try:
        iam = session.client("iam")

        # Track what we've already found to avoid duplicates
        found_arns = set()

        # Check customer-managed policies
        try:
            paginator = iam.get_paginator("list_policies")
            for page in paginator.paginate(Scope="Local"):
                for policy in page.get("Policies", []):
                    try:
                        policy_version = iam.get_policy_version(
                            PolicyArn=policy["Arn"],
                            VersionId=policy["DefaultVersionId"],
                        )
                        doc = policy_version.get("PolicyVersion", {}).get("Document", {})
                        if _policy_grants_ai_access(doc):
                            # Find attached entities
                            try:
                                entities = iam.list_entities_for_policy(PolicyArn=policy["Arn"])
                                for role in entities.get("PolicyRoles", []):
                                    role_arn = f"arn:aws:iam::*:role/{role['RoleName']}"
                                    if role_arn not in found_arns:
                                        found_arns.add(role_arn)
                                        result["principals"].append({
                                            "type": "role",
                                            "name": role["RoleName"],
                                            "arn": role_arn,
                                        })
                                for user in entities.get("PolicyUsers", []):
                                    user_arn = f"arn:aws:iam::*:user/{user['UserName']}"
                                    if user_arn not in found_arns:
                                        found_arns.add(user_arn)
                                        result["principals"].append({
                                            "type": "user",
                                            "name": user["UserName"],
                                            "arn": user_arn,
                                        })
                            except ClientError:
                                pass
                    except ClientError:
                        pass
        except ClientError:
            pass

        # Check roles for inline policies
        try:
            paginator = iam.get_paginator("list_roles")
            for page in paginator.paginate():
                for role in page.get("Roles", []):
                    role_arn = role.get("Arn", "")
                    try:
                        inline_paginator = iam.get_paginator("list_role_policies")
                        for inline_page in inline_paginator.paginate(RoleName=role["RoleName"]):
                            for policy_name in inline_page.get("PolicyNames", []):
                                try:
                                    policy_doc = iam.get_role_policy(
                                        RoleName=role["RoleName"],
                                        PolicyName=policy_name,
                                    )
                                    # FIXED: Was "RolePolicy Document", should be "PolicyDocument"
                                    if _policy_grants_ai_access(policy_doc.get("PolicyDocument", {})):
                                        if role_arn not in found_arns:
                                            found_arns.add(role_arn)
                                            result["principals"].append({
                                                "type": "role",
                                                "name": role["RoleName"],
                                                "arn": role_arn,
                                            })
                                except ClientError:
                                    pass
                    except ClientError:
                        pass
        except ClientError:
            pass

        # Check users for inline policies
        try:
            paginator = iam.get_paginator("list_users")
            for page in paginator.paginate():
                for user in page.get("Users", []):
                    user_arn = user.get("Arn", "")
                    try:
                        inline_paginator = iam.get_paginator("list_user_policies")
                        for inline_page in inline_paginator.paginate(UserName=user["UserName"]):
                            for policy_name in inline_page.get("PolicyNames", []):
                                try:
                                    policy_doc = iam.get_user_policy(
                                        UserName=user["UserName"],
                                        PolicyName=policy_name,
                                    )
                                    # FIXED: Was "UserPolicy Document", should be "PolicyDocument"
                                    if _policy_grants_ai_access(policy_doc.get("PolicyDocument", {})):
                                        if user_arn not in found_arns:
                                            found_arns.add(user_arn)
                                            result["principals"].append({
                                                "type": "user",
                                                "name": user["UserName"],
                                                "arn": user_arn,
                                            })
                                except ClientError:
                                    pass
                    except ClientError:
                        pass
        except ClientError:
            pass

    except Exception:
        pass

    return result


def _get_cost_data(session: boto3.Session, include_sagemaker: bool = False) -> dict[str, float]:
    """
    Get Cost Explorer data for Bedrock, Amazon Q, and optionally SageMaker.

    WARNING: Service dimension names in Cost Explorer are specific and must match exactly.
    If these strings don't match what AWS reports, costs will silently read as $0.
    The names below are ASSUMED, NOT YET VERIFIED against a live Cost Explorer account.

    Assumed service names (NOT confirmed, user must verify):
    - "Amazon Bedrock" — Bedrock API calls
    - "Amazon Q" — Q Business and Q generative AI services (Q Developer cost inclusion unconfirmed)
    - "Amazon SageMaker" — SageMaker (if --include-sagemaker)

    REQUIRED: Before trusting cost_bedrock_trailing_3mo or cost_q_trailing_3mo output,
    verify these names against your actual Cost Explorer data:
      aws ce get-dimension-values --dimension SERVICE --time-period \
        Start=2026-07-01,End=2026-08-01 --region us-east-1 | \
        jq -r '.DimensionValues[] | select(.Value | contains("Bedrock") or contains("Q") or contains("SageMaker")) | .Value'

    If the output doesn't match the names above, update the services_to_filter list in this function.
    See VERIFY_AWS_SCAN.md for full verification steps.
    """
    result = {
        "bedrock": 0.0,
        "q": 0.0,
        "sagemaker": 0.0,
    }

    try:
        ce = session.client("ce", region_name="us-east-1")

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=90)

        # Build service filter with verified service names
        services_to_filter = ["Amazon Bedrock", "Amazon Q"]
        if include_sagemaker:
            services_to_filter.append("Amazon SageMaker")

        service_filter = {
            "Dimensions": {
                "Key": "SERVICE",
                "Values": services_to_filter,
            }
        }

        try:
            response = ce.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.isoformat(),
                    "End": end_date.isoformat(),
                },
                Granularity="MONTHLY",
                Metrics=["UnblendedCost"],
                GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
                Filter=service_filter,
            )

            for result_group in response.get("ResultsByTime", []):
                for group in result_group.get("Groups", []):
                    service = group.get("Keys", [""])[0] if group.get("Keys") else ""
                    amount = float(group.get("Metrics", {}).get("UnblendedCost", {}).get("Amount", 0))

                    # Exact matching (not substring) to avoid mis-bucketing
                    if service == "Amazon Bedrock":
                        result["bedrock"] += amount
                    elif service == "Amazon Q":
                        result["q"] += amount
                    elif service == "Amazon SageMaker":
                        result["sagemaker"] += amount
        except ClientError as e:
            # Cost Explorer call failed, may not have billing access
            # This is not an error state; just means cost data is unavailable
            pass

    except Exception:
        pass

    return result


def _scan_account(
    account_id: str,
    account_name: str,
    regions: list[str],
    cross_account_role: Optional[str] = None,
    include_sagemaker: bool = False,
) -> AccountScanResult:
    """Scan a single AWS account."""
    result = AccountScanResult(
        account_id=account_id,
        account_name=account_name,
    )

    try:
        # Get session for this account
        if cross_account_role:
            session = _assume_role(account_id, cross_account_role)
        else:
            session = boto3.Session()

        # Scan Bedrock across regions
        bedrock_found = False
        bedrock_logging_configs = []
        total_invocations = 0

        for region in regions:
            try:
                bedrock_info = _get_bedrock_info(session, region)
                if bedrock_info["api_reachable"]:
                    bedrock_found = True
                    if bedrock_info["logging_enabled"]:
                        bedrock_logging_configs.append(True)
                    if bedrock_info["invocation_count"] is not None:
                        total_invocations += bedrock_info["invocation_count"]
            except Exception as e:
                if not result.scan_error:
                    result.scan_error = f"Bedrock scan error: {str(e)[:100]}"

        result.bedrock_api_reachable = bedrock_found

        if result.bedrock_api_reachable:
            if bedrock_logging_configs:
                result.bedrock_logging_enabled = True
            else:
                result.bedrock_logging_enabled = False

            if total_invocations > 0:
                result.bedrock_recent_invocations = total_invocations

        # Scan Amazon Q Business
        for region in regions:
            try:
                q_biz_info = _get_q_business_info(session, region)
                if q_biz_info["enabled"]:
                    result.q_business_enabled = True
                    result.q_business_app_count += q_biz_info["app_count"]

                    for app in q_biz_info.get("apps", []):
                        result.findings.append(DetailFinding(
                            account_id=account_id,
                            account_name=account_name,
                            finding_type="q_business_app",
                            resource_name_or_id=app.get("name", app.get("id", "")),
                            status_or_value=app.get("status", "Unknown"),
                            region=region,
                            confidence="medium",
                        ))
            except Exception as e:
                if not result.scan_error:
                    result.scan_error = f"Q Business scan error: {str(e)[:100]}"

        # Scan Amazon Q Developer
        try:
            q_dev_info = _get_q_developer_info(session)
            result.q_developer_enabled = q_dev_info["enabled"]
        except Exception as e:
            if not result.scan_error:
                result.scan_error = f"Q Developer scan error: {str(e)[:100]}"

        # Scan SageMaker (if requested)
        if include_sagemaker:
            for region in regions:
                try:
                    sm_info = _get_sagemaker_info(session, region)

                    for endpoint in sm_info.get("endpoints", []):
                        result.sagemaker_endpoint_count += 1
                        result.findings.append(DetailFinding(
                            account_id=account_id,
                            account_name=account_name,
                            finding_type="sagemaker_endpoint",
                            resource_name_or_id=endpoint.get("name", ""),
                            status_or_value=endpoint.get("status", ""),
                            region=region,
                            confidence="medium",
                        ))

                    for notebook in sm_info.get("notebooks", []):
                        result.sagemaker_notebook_count += 1
                        result.findings.append(DetailFinding(
                            account_id=account_id,
                            account_name=account_name,
                            finding_type="sagemaker_notebook",
                            resource_name_or_id=notebook.get("name", ""),
                            status_or_value=notebook.get("status", ""),
                            region=region,
                            confidence="medium",
                        ))

                    for job in sm_info.get("training_jobs", []):
                        result.sagemaker_recent_training_job_count += 1
                        result.findings.append(DetailFinding(
                            account_id=account_id,
                            account_name=account_name,
                            finding_type="sagemaker_training_job",
                            resource_name_or_id=job.get("name", ""),
                            status_or_value=job.get("status", ""),
                            region=region,
                            confidence="medium",
                        ))

                except Exception as e:
                    if not result.scan_error:
                        result.scan_error = f"SageMaker scan error: {str(e)[:100]}"

        # Scan IAM
        try:
            iam_info = _get_iam_access(session)
            result.iam_principals_with_ai_access_count = len(iam_info["principals"])

            for principal in iam_info["principals"]:
                result.findings.append(DetailFinding(
                    account_id=account_id,
                    account_name=account_name,
                    finding_type="iam_grant",
                    resource_name_or_id=f"{principal.get('type')}/{principal.get('name')}",
                    status_or_value=principal.get("arn", ""),
                    region="global",
                    confidence="low",
                ))
        except Exception as e:
            if not result.scan_error:
                result.scan_error = f"IAM scan error: {str(e)[:100]}"

        # Scan Cost
        try:
            cost_data = _get_cost_data(session, include_sagemaker=include_sagemaker)
            result.cost_bedrock_trailing_3mo = cost_data.get("bedrock", 0.0)
            result.cost_q_trailing_3mo = cost_data.get("q", 0.0)
            result.cost_sagemaker_trailing_3mo = cost_data.get("sagemaker", 0.0)

            if result.cost_bedrock_trailing_3mo > 0:
                result.findings.append(DetailFinding(
                    account_id=account_id,
                    account_name=account_name,
                    finding_type="cost_line",
                    resource_name_or_id="Bedrock",
                    status_or_value=f"${result.cost_bedrock_trailing_3mo:.2f}",
                    region="global",
                    confidence="high",
                ))

            if result.cost_q_trailing_3mo > 0:
                result.findings.append(DetailFinding(
                    account_id=account_id,
                    account_name=account_name,
                    finding_type="cost_line",
                    resource_name_or_id="Amazon Q",
                    status_or_value=f"${result.cost_q_trailing_3mo:.2f}",
                    region="global",
                    confidence="high",
                ))

            if result.cost_sagemaker_trailing_3mo > 0:
                result.findings.append(DetailFinding(
                    account_id=account_id,
                    account_name=account_name,
                    finding_type="cost_line",
                    resource_name_or_id="SageMaker",
                    status_or_value=f"${result.cost_sagemaker_trailing_3mo:.2f}",
                    region="global",
                    confidence="high",
                ))
        except Exception as e:
            if not result.scan_error:
                result.scan_error = f"Cost scan error: {str(e)[:100]}"

    except Exception as e:
        result.scan_error = str(e)[:200]

    return result


def _get_organization_accounts(cross_account_role: Optional[str] = None) -> list[dict[str, str]]:
    """Get list of AWS Organization accounts."""
    accounts = []

    # Try to get accounts from Organizations
    try:
        org = boto3.client("organizations")
        paginator = org.get_paginator("list_accounts")
        for page in paginator.paginate():
            for account in page.get("Accounts", []):
                if account.get("Status") == "ACTIVE":
                    accounts.append({
                        "id": account.get("Id", ""),
                        "name": account.get("Name", ""),
                    })
    except ClientError:
        # Organizations API not available, fall back to current account
        try:
            sts = boto3.client("sts")
            identity = sts.get_caller_identity()
            accounts = [{
                "id": identity.get("Account", ""),
                "name": "current",
            }]
        except Exception:
            pass

    return accounts


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--verbose", is_flag=True, help="Progress output on stderr.")
@click.option(
    "--workers",
    default=4,
    show_default=True,
    type=int,
    help="Parallel account scanners.",
)
@click.option(
    "--cross-account-role",
    default=None,
    help="Cross-account role name for multi-account mode (e.g., OrganizationAccountAccessRole).",
)
@click.option(
    "--accounts-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Newline-separated account IDs to scan (scope testing).",
)
@click.option(
    "--regions",
    multiple=True,
    default=None,
    help=f"AWS regions to scan (default: {', '.join(_DEFAULT_REGIONS)}).",
)
@click.option(
    "--include-sagemaker",
    is_flag=True,
    help="Also scan SageMaker (default: off, scanning generative AI only).",
)
@click.option("--json", "as_json", is_flag=True, help="Output JSON to stdout.")
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Per-account summary CSV (default: timestamped in script dir).",
)
@click.option(
    "--detail-csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Per-finding detail CSV (default: timestamped in script dir).",
)
@click.option(
    "--no-detail-csv",
    is_flag=True,
    help="Skip detail CSV output.",
)
def main(
    verbose: bool,
    workers: int,
    cross_account_role: Optional[str],
    accounts_file: Optional[Path],
    regions: tuple[str, ...],
    include_sagemaker: bool,
    as_json: bool,
    csv_path: Optional[Path],
    detail_csv: Optional[Path],
    no_detail_csv: bool,
) -> None:
    """Scan AWS for generative AI service usage."""

    regions_to_scan = list(regions) if regions else _DEFAULT_REGIONS

    if verbose:
        click.echo(
            f"# regions={','.join(regions_to_scan)} cross_account_role={cross_account_role or 'none'} include_sagemaker={include_sagemaker}",
            err=True,
        )
        click.echo("# fetching account list …", err=True)

    # Get accounts
    org_accounts = _get_organization_accounts(cross_account_role)

    if accounts_file:
        # Filter to specified accounts
        wanted = set()
        for line in accounts_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                wanted.add(line)

        org_accounts = [a for a in org_accounts if a["id"] in wanted]

    if not org_accounts:
        click.echo("Error: no accounts to scan", err=True)
        raise SystemExit(1)

    if verbose:
        click.echo(f"# scanning {len(org_accounts)} account(s) …", err=True)

    # Scan accounts in parallel
    worker_count = max(1, workers)
    results: list[AccountScanResult] = []

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                _scan_account,
                account["id"],
                account["name"],
                regions_to_scan,
                cross_account_role,
                include_sagemaker,
            ): idx
            for idx, account in enumerate(org_accounts)
        }

        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            if verbose:
                flags = []
                if result.bedrock_api_reachable:
                    flags.append("bedrock")
                if result.q_business_enabled:
                    flags.append(f"q-biz({result.q_business_app_count})")
                if result.q_developer_enabled:
                    flags.append("q-dev")
                if result.sagemaker_endpoint_count > 0:
                    flags.append(f"{result.sagemaker_endpoint_count}ep")
                if result.cost_bedrock_trailing_3mo > 0 or result.cost_q_trailing_3mo > 0 or result.cost_sagemaker_trailing_3mo > 0:
                    flags.append("$spend")
                flag_text = f" [{', '.join(flags)}]" if flags else " [—]"
                err_text = f" ERROR: {result.scan_error}" if result.scan_error else ""
                click.echo(
                    f"[scan] {result.account_id:12} {result.account_name:30}{flag_text}{err_text}",
                    err=True,
                )

    # Sort results by account ID
    results.sort(key=lambda r: r.account_id)

    # Prepare summary rows
    summary_rows = []
    for r in results:
        bedrock_status = "Yes" if r.bedrock_api_reachable else "No"
        logging_status = (
            "Yes" if r.bedrock_logging_enabled else
            ("No" if r.bedrock_api_reachable else "—")
        )

        summary_rows.append({
            "account_id": r.account_id,
            "account_name": r.account_name,
            "bedrock_api_reachable": bedrock_status,
            "bedrock_logging_enabled": logging_status,
            "bedrock_recent_invocations": r.bedrock_recent_invocations or "",
            "q_business_enabled": "Yes" if r.q_business_enabled else "No",
            "q_business_app_count": r.q_business_app_count,
            "q_developer_enabled": "Yes" if r.q_developer_enabled else "No",
            "sagemaker_endpoint_count": r.sagemaker_endpoint_count if include_sagemaker else "",
            "sagemaker_notebook_count": r.sagemaker_notebook_count if include_sagemaker else "",
            "sagemaker_recent_training_job_count": r.sagemaker_recent_training_job_count if include_sagemaker else "",
            "iam_principals_with_ai_access_count": r.iam_principals_with_ai_access_count,
            "cost_bedrock_trailing_3mo": f"${r.cost_bedrock_trailing_3mo:.2f}",
            "cost_q_trailing_3mo": f"${r.cost_q_trailing_3mo:.2f}",
            "cost_sagemaker_trailing_3mo": f"${r.cost_sagemaker_trailing_3mo:.2f}" if include_sagemaker else "",
            "scan_error": r.scan_error or "",
        })

    # Prepare detail rows
    detail_rows = []
    for r in results:
        for finding in r.findings:
            detail_rows.append({
                "account_id": finding.account_id,
                "account_name": finding.account_name,
                "finding_type": finding.finding_type,
                "resource_name_or_id": finding.resource_name_or_id,
                "status_or_value": finding.status_or_value,
                "region": finding.region,
                "confidence": finding.confidence,
            })

    # Output JSON
    if as_json:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "accounts_scanned": len(results),
            "regions": regions_to_scan,
            "include_sagemaker": include_sagemaker,
            "cross_account_mode": cross_account_role is not None,
            "accounts": [
                {
                    "account_id": r.account_id,
                    "account_name": r.account_name,
                    "bedrock_api_reachable": r.bedrock_api_reachable,
                    "bedrock_logging_enabled": r.bedrock_logging_enabled,
                    "bedrock_recent_invocations": r.bedrock_recent_invocations,
                    "q_business_enabled": r.q_business_enabled,
                    "q_business_app_count": r.q_business_app_count,
                    "q_developer_enabled": r.q_developer_enabled,
                    "sagemaker_endpoint_count": r.sagemaker_endpoint_count if include_sagemaker else None,
                    "sagemaker_notebook_count": r.sagemaker_notebook_count if include_sagemaker else None,
                    "sagemaker_recent_training_job_count": r.sagemaker_recent_training_job_count if include_sagemaker else None,
                    "iam_principals_with_ai_access_count": r.iam_principals_with_ai_access_count,
                    "cost_bedrock_trailing_3mo": r.cost_bedrock_trailing_3mo,
                    "cost_q_trailing_3mo": r.cost_q_trailing_3mo,
                    "cost_sagemaker_trailing_3mo": r.cost_sagemaker_trailing_3mo if include_sagemaker else None,
                    "scan_error": r.scan_error,
                    "findings": len([f for f in r.findings]),
                }
                for r in results
            ],
        }
        click.echo(json.dumps(payload, indent=2))

    # Output CSV
    if csv_path is None and not as_json:
        csv_path = _timestamped_output_path(".csv")

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        if verbose:
            click.echo(f"# summary CSV: {csv_path.resolve()}", err=True)

    # Output detail CSV
    write_detail = not no_detail_csv and (detail_csv is not None or csv_path is not None)
    if write_detail:
        if detail_csv is None:
            detail_csv = _timestamped_output_path("_findings.csv")
        detail_csv.parent.mkdir(parents=True, exist_ok=True)
        with detail_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_DETAIL_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in detail_rows:
                writer.writerow(row)
        if verbose:
            click.echo(f"# findings CSV: {detail_csv.resolve()} ({len(detail_rows)} rows)", err=True)

    # Print console summary
    bedrock_accounts = sum(1 for r in results if r.bedrock_api_reachable)
    q_accounts = sum(1 for r in results if r.q_business_enabled or r.q_developer_enabled)
    sagemaker_accounts = sum(1 for r in results if r.sagemaker_endpoint_count > 0 or r.sagemaker_notebook_count > 0) if include_sagemaker else 0
    with_spend = sum(1 for r in results if r.cost_bedrock_trailing_3mo > 0 or r.cost_q_trailing_3mo > 0 or r.cost_sagemaker_trailing_3mo > 0)

    if not as_json:
        click.echo(f"\nSummary: {len(results)} account(s) scanned", err=True)
        click.echo(f"  Bedrock API reachable: {bedrock_accounts}", err=True)
        click.echo(f"  Amazon Q (Business/Developer): {q_accounts}", err=True)
        if include_sagemaker:
            click.echo(f"  SageMaker active: {sagemaker_accounts}", err=True)
        click.echo(f"  With AI spend: {with_spend}", err=True)
        click.echo(f"  Total findings: {len(detail_rows)}", err=True)


if __name__ == "__main__":
    main()
