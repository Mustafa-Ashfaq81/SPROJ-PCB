import subprocess
import shlex
import sys
import os

def run_command(command):
    """
    Runs a shell command in real-time, printing output to terminal line-by-line.
    Raises RuntimeError if command fails.
    """
    print(f"\n[INFO] Running command:\n{command}\n{'-'*60}")

    parts = shlex.split(command)

    if parts[0] == "python":
        parts.insert(1, "-u")  # Force unbuffered mode for real-time logs

    process = subprocess.Popen(
        parts,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line-buffered
    )

    for line in iter(process.stdout.readline, ''):
        if line:
            print(line, end='')

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Command failed (exit code={return_code}):\n{command}")

def main():
    commands_file = "marnus_labuschagne_commands_with_models.txt"

    if not os.path.exists(commands_file):
        print(f"[ERROR] File not found: {commands_file}")
        sys.exit(1)

    with open(commands_file, "r", encoding="utf-8") as f:
        commands = [line.strip() for line in f if line.strip()]

    for idx, cmd in enumerate(commands, 1):
        try:
            print(f"\n=== Running Command {idx}/{len(commands)} ===")
            run_command(cmd)
        except Exception as e:
            print(f"\n[ERROR] Failed at command {idx}: {e}")
            sys.exit(1)

    print("\n✅ All commands executed successfully.")

if __name__ == "__main__":
    main()
