import os
import subprocess
import argparse

PBS_TEMPLATE = """#!/bin/bash
#PBS -N FATES_calib_{pft_id}
#PBS -q {queue}
#PBS -l select=1:ncpus={ncpus}:mpiprocs={mpi}:mem={mem}
#PBS -l walltime={walltime}
#PBS -A {project}
#PBS -j oe
#PBS -k eod
#PBS -m abe
#PBS -e {job_dir}/error_{pft_id}.txt
#PBS -o {job_dir}/output_{pft_id}.txt
#PBS -M {email}

module load conda
conda activate fates_calibration

cd {script_dir}
mpiexec -n {mpi} python {script_name} --pft {pft_id} --bootstraps {bootstraps} --out-dir {out_dir} --config {config}
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Generate PBS job scripts for FATES calibration.\n")
    parser.add_argument("--pfts", nargs="+", required=True,
                        help="List of PFTs to run (e.g. 1 2 3 4)\n")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file\n")
    parser.add_argument("--bootstraps", type=int, default=100,
                        help="Number of bootstrap runs per PFT\n")
    parser.add_argument("--script", type=str, default="calibrate_scipy.py",
                        help="Python script to execute\n")
    parser.add_argument("--script-dir", type=str, default="/glade/work/afoster/FATES_calibration/fates_calibration_library/scripts",
                        help="Path to directory containing the script\n")
    parser.add_argument("--output-base", type=str, default="/glade/work/afoster/FATES_calibration/param_outputs_history_matching",
                        help="Base output directory for PFT outputs\n")
    parser.add_argument("--job-dir", type=str, default="/glade/work/afoster/FATES_calibration/jobs",
                        help="Directory to write PBS job scripts\n")

    # PBS/Compute resource options
    parser.add_argument("--walltime", type=str, default="12:00:00",
                        help="Walltime (HH:MM:SS)\n")
    parser.add_argument("--mem", type=str, default="100G",
                        help="Memory allocation\n")
    parser.add_argument("--mpi", type=int, default=2,
                        help="Number of MPI processes\n")
    parser.add_argument("--queue", type=str, default="casper",
                        help="PBS queue to submit to\n")
    parser.add_argument("--project", type=str, default="P93300041",
                        help="Project account code\n")
    parser.add_argument("--email", type=str, default="afoster@ucar.edu",
                        help="Email for job notifications\n")

    # Flags
    parser.add_argument("--submit", action="store_true",
                        help="Submit jobs to the queue")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print actions, don't write or submit")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.dry_run:
        os.makedirs(args.job_dir, exist_ok=True)
    script_path = os.path.join(args.script_dir, args.script)
    if not os.path.isfile(script_path):
        raise IOError(f"ERROR: Script not found: {script_path}")
    
    for pft_id in args.pfts:
        output_dir = os.path.join(args.output_base, f"{pft_id}_outputs")
        job_file = os.path.join(args.job_dir, f"calib_{pft_id}.pbs")

        job_content = PBS_TEMPLATE.format(
            pft_id=pft_id,
            queue=args.queue,
            walltime=args.walltime,
            mem=args.mem,
            ncpus=args.mpi,
            mpi=args.mpi,
            project=args.project,
            email=args.email,
            job_dir=args.job_dir,
            script_dir=args.script_dir,
            script_name=args.script,
            bootstraps=args.bootstraps,
            out_dir=output_dir,
            config=args.config
        )

        if args.dry_run:
            print(f"[Dry Run] Would generate and submit job for: {pft_id}")
            print(f"Script Path: {script_path}")
            print(f"Output Dir: {output_dir}")
            print("--- PBS Script ---")
            print(job_content)
            print("------------------\n")
        else:
            os.makedirs(output_dir, exist_ok=True)
            with open(job_file, "w", encoding="utf-8") as f:
                f.write(
                    job_content
                )
            print(f"Written: {job_file}")
            if args.submit:
                print(f"Submitting {job_file}")
                subprocess.run(["qsub", str(job_file)])

if __name__ == "__main__":
    main()