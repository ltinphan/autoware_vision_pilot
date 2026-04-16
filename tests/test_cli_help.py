import subprocess
import sys

def test_cli_help():
    rc = subprocess.run([sys.executable, 'Models/inference/cli.py', '--help'], capture_output=True)
    assert rc.returncode == 0
    out = rc.stdout.decode('utf-8') + rc.stderr.decode('utf-8')
    assert '--model' in out
    assert '--checkpoint' in out
