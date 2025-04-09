"""Trains many models through disBatch in parallel"""

from pathlib import Path

import numpy as np

transition_probs = np.array(
    [
        [0.95, 0.05],
        [0.05, 0.95],
    ]
)

emission_probs = np.array(
    [
        [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
        [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 2],
    ]
)

initial_state_probs = np.array([1 / 2, 1 / 2])

source_envs_cmd = "source /mnt/home/atanelus/.bashrc; source /mnt/home/atanelus/venvs/new/bin/activate"


def write_hmm_params(experiment_dir: Path) -> None:
    np.savez(
        experiment_dir / "hmm_params.npz",
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        initial_probs=initial_state_probs,
    )


def write_batch_script(
    experiment_dir: Path,
    training_set_sizes: list[int],
    num_iters: int,
    batch_size: int,
    seq_len: int,
):
    # Find the train_model script
    train_model_script = Path(__file__).parent / "train_model.py"
    train_model_script = train_model_script.absolute()
    if not train_model_script.exists():
        raise FileNotFoundError(f"Cannot find train_model.py")

    log_dir = experiment_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    model_dir = experiment_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(experiment_dir / "train_disbatch", "w") as ctx:
        for model_type in ("rnn", "transformer"):
            for train_size in training_set_sizes:
                model_save_path = model_dir / f"{model_type}_{train_size}_samples"
                log_path = log_dir / f"{model_type}_{train_size}_samples.log"
                cmd_format = (
                    f"python {train_model_script} "
                    f"--hmm-params-file {experiment_dir / 'hmm_params.npz'} "
                    f"--num-iters {num_iters} "
                    f"--seq-len {seq_len} "
                    "--model {model_type} "
                    "--output-dir {model_save_path} "
                    "--train-size {train_size}"
                )
                cmd = cmd_format.format(
                    model_type=model_type,
                    model_save_path=model_save_path,
                    train_size=train_size,
                )
                cmd = f"( {source_envs_cmd}; {cmd} ) &> {log_path}\n"
                ctx.write(cmd)

    num_jobs = len(training_set_sizes) * 2
    num_jobs = min(num_jobs, 20)

    print(
        f"Writing batch script to {experiment_dir / 'train_disbatch'} with {num_jobs} jobs"
    )
    print("To run:")
    print(
        f"module load disBatch; mkdir trash; sbatch -p gpu -n {num_jobs} -t 0-6 -c 12 --mem=48GB --gpus-per-task=1 disBatch -p trash/ {experiment_dir / 'train_disbatch'}"
    )


if __name__ == "__main__":
    exp_dir = Path("/mnt/home/atanelus/ceph/artificial_grammars_sweep")
    write_hmm_params(exp_dir)
    write_batch_script(
        experiment_dir=exp_dir,
        training_set_sizes=[500, 1000, 5000, 10000, 50000, 100000],
        num_iters=100000,
        batch_size=32,
        seq_len=512,
    )
