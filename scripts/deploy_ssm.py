"""Send the existing deployment script through SSM and wait for its real exit status."""
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

SCRIPT = Path(__file__).with_name('deploy_ec2.sh')


def aws(*arguments):
    result = subprocess.run(
        ['aws', 'ssm', *arguments, '--output', 'json', '--no-cli-pager'],
        check=True, capture_output=True, text=True, timeout=90,
    )
    return json.loads(result.stdout or '{}')


def command_payload(instance, image):
    if not re.fullmatch(r'i-(?:[a-f0-9]{8}|[a-f0-9]{17})', instance):
        raise ValueError('EC2_INSTANCE_ID must be a valid EC2 instance ID')
    if not re.fullmatch(r'ghcr\.io/[a-z0-9._/-]+@sha256:[a-f0-9]{64}', image):
        raise ValueError('IMAGE must be a GHCR image pinned by SHA256 digest')
    encoded = base64.b64encode(SCRIPT.read_bytes()).decode('ascii')
    # Encode the script to preserve its Bash syntax through JSON and SSM's shell.
    commands = '\n'.join([
        'set -eu',
        'deploy_script=$(mktemp)',
        "trap 'rm -f \"$deploy_script\"' EXIT",
        f"printf '%s' '{encoded}' | base64 --decode > \"$deploy_script\"",
        f'bash "$deploy_script" \'{image}\'',
    ])
    return {
        'InstanceIds': [instance], 'DocumentName': 'AWS-RunShellScript',
        'Comment': 'Deploy MemoryBot published image', 'TimeoutSeconds': 60,
        'Parameters': {'commands': [commands], 'executionTimeout': ['900']},
    }


def deploy(instance, image, wait_seconds=1200):
    response = aws('send-command', '--cli-input-json', json.dumps(command_payload(instance, image)))
    command_id = response['Command']['CommandId']
    print(f'SSM command submitted: {command_id}', flush=True)
    deadline = time.monotonic() + wait_seconds
    terminal = False
    try:
        while time.monotonic() < deadline:
            try:
                result = aws('get-command-invocation', '--command-id', command_id,
                             '--instance-id', instance)
            except subprocess.CalledProcessError as error:
                # SSM is eventually consistent immediately after SendCommand.
                if 'InvocationDoesNotExist' not in (error.stderr or ''):
                    raise
                time.sleep(10)
                continue
            status = result['Status']
            if status in ('Success', 'Failed', 'Cancelled', 'TimedOut'):
                terminal = True
                print(result.get('StandardOutputContent', ''), flush=True)
                print(result.get('StandardErrorContent', ''), file=sys.stderr, flush=True)
                if status != 'Success' or result.get('ResponseCode') != 0:
                    raise RuntimeError(f'SSM deployment failed: {status}')
                print('MemoryBot deployment succeeded', flush=True)
                return
            if status not in ('Pending', 'InProgress', 'Delayed', 'Cancelling'):
                raise RuntimeError(f'Unexpected SSM status: {status}')
            time.sleep(10)
        raise TimeoutError('Timed out waiting for deployment; inspect SSM command history')
    finally:
        if not terminal:
            # Cancellation is best effort; interrupted remote processes may need inspection.
            try:
                aws('cancel-command', '--command-id', command_id, '--instance-ids', instance)
            except Exception as error:
                print(f'Could not cancel SSM command {command_id}: {error}', file=sys.stderr)


if __name__ == '__main__':
    try:
        deploy(os.environ['EC2_INSTANCE_ID'], os.environ['IMAGE'])
    except Exception as error:
        print(f'Deployment failed: {error}', file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(error.stderr or '', file=sys.stderr)
        sys.exit(1)
