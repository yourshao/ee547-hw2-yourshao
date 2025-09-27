import argparse
import sys
import json
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError

STD_RETRY_CONFIG = Config(retries={'max_attempts': 2, 'mode': 'standard'}, connect_timeout=5, read_timeout=10)

def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def validate_region(region: Optional[str]) -> Optional[str]:
    if not region:
        return None
    available = boto3.session.Session().get_available_regions('ec2')
    if region not in available:
        raise ValueError(
            f"Invalid region '{region}'. "
            f"Valid examples include: {', '.join(sorted(set(available))[:8])} ..."
        )
    return region

def get_sts_identity(session: boto3.session.Session) -> Dict[str, Any]:
    sts = session.client('sts', config=STD_RETRY_CONFIG)
    return sts.get_caller_identity()

def collect_iam_users(session: boto3.session.Session) -> List[Dict[str, Any]]:
    iam = session.client('iam', config=STD_RETRY_CONFIG)
    users: List[Dict[str, Any]] = []
    try:
        paginator = iam.get_paginator('list_users')
        for page in paginator.paginate():
            for u in page.get('Users', []):
                username = u.get('UserName')
                user_record: Dict[str, Any] = {
                    "username": username,
                    "user_id": u.get('UserId'),
                    "arn": u.get('Arn'),
                    "create_date": u.get('CreateDate').strftime('%Y-%m-%dT%H:%M:%SZ') if u.get('CreateDate') else None,
                    "last_activity": None,
                    "attached_policies": []
                }
                # Try to fetch PasswordLastUsed via GetUser
                try:
                    gu = iam.get_user(UserName=username).get('User', {})
                    plu = gu.get('PasswordLastUsed')
                    if plu:
                        user_record["last_activity"] = plu.strftime('%Y-%m-%dT%H:%M:%SZ')
                except ClientError as ce:
                    code = ce.response.get('Error', {}).get('Code', 'Unknown')
                    eprint(f"[WARNING] iam:GetUser failed for '{username}' ({code}) - continuing")

                # Attached managed policies
                try:
                    ap = iam.list_attached_user_policies(UserName=username).get('AttachedPolicies', [])
                    user_record["attached_policies"] = [
                        {"policy_name": p.get('PolicyName'), "policy_arn": p.get('PolicyArn')} for p in ap
                    ]
                except ClientError as ce:
                    code = ce.response.get('Error', {}).get('Code', 'Unknown')
                    eprint(f"[WARNING] iam:ListAttachedUserPolicies failed for '{username}' ({code}) - continuing")

                users.append(user_record)
    except ClientError as ce:
        code = ce.response.get('Error', {}).get('Code', 'Unknown')
        eprint(f"[WARNING] Access denied for IAM operations ({code}) - skipping user enumeration")
    except EndpointConnectionError:
        eprint("[WARNING] Network timeout on IAM - retried once, skipping")
    return users

def collect_ec2(session: boto3.session.Session, region: Optional[str]) -> List[Dict[str, Any]]:
    try:
        ec2 = session.client('ec2', region_name=region, config=STD_RETRY_CONFIG)
    except Exception:
        ec2 = session.client('ec2', config=STD_RETRY_CONFIG)

    instances: List[Dict[str, Any]] = []
    try:
        paginator = ec2.get_paginator('describe_instances')
        reservations = []
        for page in paginator.paginate():
            reservations.extend(page.get('Reservations', []))

        # Collect distinct AMI IDs for batch lookup
        ami_ids = set()
        for r in reservations:
            for inst in r.get('Instances', []):
                if inst.get('ImageId'):
                    ami_ids.add(inst['ImageId'])

        ami_name_map: Dict[str, str] = {}
        if ami_ids:
            try:
                ids = list(ami_ids)
                for i in range(0, len(ids), 100):
                    resp = ec2.describe_images(ImageIds=ids[i:i+100])
                    for img in resp.get('Images', []):
                        ami_name_map[img['ImageId']] = img.get('Name')
            except ClientError as ce:
                code = ce.response.get('Error', {}).get('Code', 'Unknown')
                eprint(f"[WARNING] ec2:DescribeImages failed ({code}) - AMI names will be blank")

        for r in reservations:
            for inst in r.get('Instances', []):
                tags = {t.get('Key'): t.get('Value') for t in (inst.get('Tags') or [])}
                sgs = [sg.get('GroupId') for sg in inst.get('SecurityGroups', [])]
                launch_time = inst.get('LaunchTime')
                instances.append({
                    "instance_id": inst.get('InstanceId'),
                    "instance_type": inst.get('InstanceType'),
                    "state": (inst.get('State') or {}).get('Name'),
                    "public_ip": inst.get('PublicIpAddress'),
                    "private_ip": inst.get('PrivateIpAddress'),
                    "availability_zone": (inst.get('Placement') or {}).get('AvailabilityZone'),
                    "launch_time": launch_time.strftime('%Y-%m-%dT%H:%M:%SZ') if launch_time else None,
                    "ami_id": inst.get('ImageId'),
                    "ami_name": ami_name_map.get(inst.get('ImageId')),
                    "security_groups": sgs,
                    "tags": tags
                })
    except ClientError as ce:
        code = ce.response.get('Error', {}).get('Code', 'Unknown')
        if code in ("AuthFailure", "UnauthorizedOperation", "AccessDenied", "AccessDeniedException"):
            eprint(f"[WARNING] Access denied for EC2 operations - skipping instances")
        else:
            eprint(f"[WARNING] EC2 describe failed ({code}) - skipping instances")
    except EndpointConnectionError:
        eprint("[WARNING] Network timeout on EC2 - retried once, skipping")
    return instances

def _sg_rule_pretty(rule: Dict[str, Any]) -> Tuple[str, str, str]:
    ip_protocol = rule.get('IpProtocol', '-1')
    protocol = 'all' if ip_protocol in ('-1', None) else ip_protocol
    from_p = rule.get('FromPort')
    to_p = rule.get('ToPort')
    if from_p is None and to_p is None and protocol == 'all':
        port_range = 'all'
    else:
        if from_p is None: from_p = to_p
        if to_p is None: to_p = from_p
        port_range = f"{from_p}-{to_p}"
    cidrs = [c.get('CidrIp') for c in rule.get('IpRanges', []) if c.get('CidrIp')]
    cidr6 = [c.get('CidrIpv6') for c in rule.get('Ipv6Ranges', []) if c.get('CidrIpv6')]
    src = ','.join(cidrs + cidr6) if (cidrs or cidr6) else None
    if src is None:
        pairs = [p.get('GroupId') for p in rule.get('UserIdGroupPairs', []) if p.get('GroupId')]
        src = ','.join(pairs) if pairs else 'N/A'
    return protocol, port_range, src

def collect_security_groups(session: boto3.session.Session, region: Optional[str]) -> List[Dict[str, Any]]:
    try:
        ec2 = session.client('ec2', region_name=region, config=STD_RETRY_CONFIG)
    except Exception:
        ec2 = session.client('ec2', config=STD_RETRY_CONFIG)
    sgs: List[Dict[str, Any]] = []
    try:
        paginator = ec2.get_paginator('describe_security_groups')
        for page in paginator.paginate():
            for sg in page.get('SecurityGroups', []):
                inbound = []
                for rule in sg.get('IpPermissions', []):
                    proto, prange, src = _sg_rule_pretty(rule)
                    inbound.append({"protocol": proto, "port_range": prange, "source": src})
                outbound = []
                for rule in sg.get('IpPermissionsEgress', []):
                    proto, prange, dest = _sg_rule_pretty(rule)
                    outbound.append({"protocol": proto, "port_range": prange, "destination": dest})
                sgs.append({
                    "group_id": sg.get('GroupId'),
                    "group_name": sg.get('GroupName'),
                    "description": sg.get('Description'),
                    "vpc_id": sg.get('VpcId'),
                    "inbound_rules": inbound,
                    "outbound_rules": outbound
                })
    except ClientError as ce:
        code = ce.response.get('Error', {}).get('Code', 'Unknown')
        if code in ("AuthFailure", "UnauthorizedOperation", "AccessDenied", "AccessDeniedException"):
            eprint(f"[WARNING] Access denied for EC2:DescribeSecurityGroups - skipping security groups")
        else:
            eprint(f"[WARNING] DescribeSecurityGroups failed ({code}) - skipping")
    except EndpointConnectionError:
        eprint("[WARNING] Network timeout on SGs - retried once, skipping")
    return sgs

def collect_s3(session: boto3.session.Session, region_for_default: Optional[str]) -> List[Dict[str, Any]]:
    s3 = session.client('s3', config=STD_RETRY_CONFIG)
    buckets_info: List[Dict[str, Any]] = []
    try:
        resp = s3.list_buckets()
        buckets = resp.get('Buckets', [])
        if not buckets:
            eprint("[WARNING] No S3 buckets found")
        for b in buckets:
            name = b.get('Name')
            creation = b.get('CreationDate')
            # Region: get-bucket-location can return None for us-east-1
            try:
                loc = s3.get_bucket_location(Bucket=name).get('LocationConstraint')
                region = loc if loc is not None else 'us-east-1'
            except ClientError as ce:
                code = ce.response.get('Error', {}).get('Code', 'Unknown')
                eprint(f"[ERROR] Failed to access S3 bucket '{name}': {code}")
                continue
            # Count objects & size (approximate by listing)
            count = 0
            size_bytes = 0
            try:
                paginator = s3.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=name):
                    for obj in page.get('Contents', []) or []:
                        count += 1
                        size_bytes += int(obj.get('Size') or 0)
            except ClientError as ce:
                code = ce.response.get('Error', {}).get('Code', 'Unknown')
                eprint(f"[ERROR] Failed to list objects in bucket '{name}': {code}")
            except EndpointConnectionError:
                eprint(f"[WARNING] Network timeout on listing bucket '{name}' - partial/zero counts")
            buckets_info.append({
                "bucket_name": name,
                "creation_date": creation.strftime('%Y-%m-%dT%H:%M:%SZ') if creation else None,
                "region": region,
                "object_count": count,
                "size_bytes": size_bytes
            })
    except ClientError as ce:
        code = ce.response.get('Error', {}).get('Code', 'Unknown')
        if code in ("AccessDenied", "AccessDeniedException"):
            eprint("[WARNING] Access denied for S3:ListAllMyBuckets - skipping buckets")
        else:
            eprint(f"[WARNING] S3 list_buckets failed ({code}) - skipping buckets")
    except EndpointConnectionError:
        eprint("[WARNING] Network timeout on S3 - retried once, skipping")
    return buckets_info

def to_table(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    acct = report["account_info"]["account_id"]
    region = report["account_info"]["region"] or "(from profile)"
    scan_ts = report["account_info"]["scan_timestamp"]
    lines.append(f"AWS Account: {acct} ({region})")
    lines.append(f"Scan Time: {scan_ts.replace('T', ' ').replace('Z',' UTC')}")
    lines.append("")

    users = report["resources"]["iam_users"]
    lines.append(f"IAM USERS ({len(users)} total)")
    lines.append(f"{'Username':20} {'Create Date':15} {'Last Activity':15} {'Policies':8}")
    for u in users:
        cd = (u.get('create_date') or '')[:10]
        la = (u.get('last_activity') or '')[:10]
        pol_cnt = len(u.get('attached_policies') or [])
        lines.append(f"{(u.get('username') or ''):20} {cd:15} {la:15} {pol_cnt:<8}")
    if not users:
        lines.append("(none)")
    lines.append("")

    inst = report["resources"]["ec2_instances"]
    run_cnt = sum(1 for i in inst if i.get('state') == 'running')
    stop_cnt = sum(1 for i in inst if i.get('state') == 'stopped')
    lines.append(f"EC2 INSTANCES ({run_cnt} running, {stop_cnt} stopped)")
    lines.append(f"{'Instance ID':20} {'Type':10} {'State':10} {'Public IP':16} {'Launch Time':19}")
    for i in inst:
        lt = (i.get('launch_time') or '').replace('T', ' ')[:-1]
        pip = i.get('public_ip') or '-'
        lines.append(f"{(i.get('instance_id') or ''):20} {(i.get('instance_type') or ''):10} {(i.get('state') or ''):10} {pip:16} {lt:19}")
    if not inst:
        lines.append("(none)")
    lines.append("")

    buckets = report["resources"]["s3_buckets"]
    lines.append(f"S3 BUCKETS ({len(buckets)} total)")
    lines.append(f"{'Bucket Name':24} {'Region':12} {'Created':12} {'Objects':8} {'Size (MB)':10}")
    for b in buckets:
        created = (b.get('creation_date') or '')[:10]
        size_mb = "~{:.1f}".format((b.get('size_bytes') or 0)/1024/1024)
        lines.append(f"{(b.get('bucket_name') or ''):24} {(b.get('region') or ''):12} {created:12} {str(b.get('object_count') or 0):8} {size_mb:10}")
    if not buckets:
        lines.append("(none)")
    lines.append("")

    sgs = report["resources"]["security_groups"]
    lines.append(f"SECURITY GROUPS ({len(sgs)} total)")
    lines.append(f"{'Group ID':14} {'Name':14} {'VPC ID':14} {'Inbound Rules'}")
    for sg in sgs:
        in_cnt = len(sg.get('inbound_rules') or [])
        lines.append(f"{(sg.get('group_id') or ''):14} {(sg.get('group_name') or ''):14} {(sg.get('vpc_id') or ''):14} {in_cnt}")
    if not sgs:
        lines.append("(none)")
    return "\n".join(lines)

def build_report(session: boto3.session.Session, region: Optional[str]) -> Dict[str, Any]:
    try:
        ident = get_sts_identity(session)
    except ClientError as ce:
        code = ce.response.get('Error', {}).get('Code', 'Unknown')
        eprint(f"[ERROR] Authentication failed (sts:GetCallerIdentity): {code}")
        sys.exit(1)
    except Exception as ex:
        eprint(f"[ERROR] Authentication failed: {ex}")
        sys.exit(1)

    acct_region = region or session.region_name
    iam_users = collect_iam_users(session)
    ec2_instances = collect_ec2(session, region)
    s3_buckets = collect_s3(session, acct_region)
    security_groups = collect_security_groups(session, region)

    report = {
        "account_info": {
            "account_id": ident.get('Account'),
            "user_arn": ident.get('Arn'),
            "region": acct_region,
            "scan_timestamp": utc_now_iso(),
        },
        "resources": {
            "iam_users": iam_users,
            "ec2_instances": ec2_instances,
            "s3_buckets": s3_buckets,
            "security_groups": security_groups,
        },
        "summary": {
            "total_users": len(iam_users),
            "running_instances": sum(1 for i in ec2_instances if i.get('state') == 'running'),
            "total_buckets": len(s3_buckets),
            "security_groups": len(security_groups),
        }
    }
    return report

def main():
    parser = argparse.ArgumentParser(description="List and inspect AWS resources across your account")
    parser.add_argument("--region", help="AWS region to inspect (default: from credentials/config)")
    parser.add_argument("--output", help="Output file path (default: stdout)")
    parser.add_argument("--format", choices=["json", "table"], default="json", help="Output format")
    args = parser.parse_args()

    try:
        valid_region = validate_region(args.region) if args.region else None
    except ValueError as ve:
        eprint(str(ve))
        sys.exit(2)

    session = boto3.session.Session(region_name=valid_region)

    report = build_report(session, valid_region)

    if args.format == "json":
        out = json.dumps(report, indent=2)
    else:
        out = to_table(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        eprint("\n[INFO] Interrupted by user")
        sys.exit(130)
