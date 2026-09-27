import subprocess
import sys


def run_step(title, command):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\n❌ {title} failed.")
        sys.exit(result.returncode)

    print(f"\n✅ {title} completed successfully.")


def main():
    run_step(
        "STEP 1 — VALIDATING EVALUATION DATASET",
        [sys.executable, "-m", "evaluation.validate_dataset"],
    )

    run_step(
        "STEP 2 — RUNNING RETRIEVAL EVALUATION",
        [sys.executable, "-m", "evaluation.run_evaluation"],
    )

    run_step(
        "STEP 3 — RUNNING GENERATION EVALUATION",
        [sys.executable, "-m", "evaluation.run_generation_evaluation"],
    )

    run_step(
        "STEP 4 — RUNNING APPLICATION TESTS",
        [sys.executable, "-m", "pytest", "tests", "-q"],
    )

    run_step(
        "STEP 5 — RUNNING EVALUATION TESTS",
        [sys.executable, "-m", "pytest", "evaluation/tests", "-q"],
    )

    print("\n" + "=" * 70)
    print("PROJECT 2 — COMPLETE EVALUATION FINISHED")
    print("=" * 70)
    print("\nAll evaluation stages completed successfully.")


if __name__ == "__main__":
    main()