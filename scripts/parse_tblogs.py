import argparse
import os

from tensorboard.backend.event_processing import event_accumulator


def main(log_file: str):
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return

    ea = event_accumulator.EventAccumulator(
        log_file,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    if "metrics/smk_loss" in ea.Tags()["scalars"]:
        smk_loss_events = ea.Scalars("metrics/smk_loss")
        print("Step | SMK Loss")
        print("-----|----------")
        for event in smk_loss_events:
            print(f"{event.step:4d} | {event.value:.6f}")
    else:
        print("SMK Loss not found in logs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse a TensorBoard event file and print smk_loss."
    )
    parser.add_argument(
        "log_file",
        type=str,
        help="Path to the TensorBoard event file.",
    )
    args = parser.parse_args()
    main(args.log_file)
