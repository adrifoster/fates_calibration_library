#!/bin/bash

job_dir=jobs
if [ ! -d "${job_dir}" ]; then
  mkdir ${job_dir}
fi

# dominant pfts
pft_ids=('BETT' 'NEET' 'NCET' 'AC3G' 'C4G' 'C3G')

# codominant pfts
#pft_ids=('BCES' 'BDET' 'BEET' 'BDTT' 'C3C' 'C3CI')

for pft_id in ${pft_ids[@]}; do

    if [ -f "${job_dir}/calib_${pft_id}" ]; then
      rm ${job_dir}/calib_${pft_id}
    fi

    pft_dir=/glade/u/home/afoster/FATES_Calibration/pft_output_gs1/${pft_id}_outputs/
    sample_dir=${pft_dir}/samples
    
    if [ ! -d "${pft_dir}" ]; then
      mkdir ${pft_dir}
    fi
    if [ ! -d "${sample_dir}" ]; then
      mkdir ${sample_dir}
    fi
    
    echo "#!/bin/bash" >> ${job_dir}/calib_${pft_id}
    echo " " >> ${job_dir}/calib_${pft_id}
    echo "#PBS -N FATES_calib_${pft_id}" >> ${job_dir}/calib_${pft_id}
    echo "#PBS -q casper" >> ${job_dir}/calib_${pft_id}
    echo "#PBS -l select=1:ncpus=16:mpiprocs=16:mem=50G" >> ${job_dir}/calib_${pft_id}
    echo "#PBS -l walltime=12:00:00" >> ${job_dir}/calib_${pft_id}
    echo "#PBS -A P93300041" >> ${job_dir}/calib_${pft_id}
    echo "#PBS -j oe" >> ${job_dir}/calib_${pft_id}
    echo "#PBS -k eod" >> ${job_dir}/calib_${pft_id}
    echo "#PBS -m abe" >> ${job_dir}/calib_${pft_id}
    echo "#PBS -e ${job_dir}/error_${pft_id}.txt" >> ${job_dir}/calib_${pft_id}
    echo "#PBS -o ${job_dir}/output_${pft_id}.txt" >> ${job_dir}/calib_${pft_id}
    echo "#PBS -M afoster@ucar.edu" >> ${job_dir}/calib_${pft_id}
    echo " " >> ${job_dir}/calib_${pft_id}
    echo "module load conda" >> ${job_dir}/calib_${pft_id}
    echo "conda activate fates_calibration" >> ${job_dir}/calib_${pft_id}
    echo " " >> ${job_dir}/calib_${pft_id}
    echo "cd /glade/u/home/afoster/FATES_Calibration/scripts" >> ${job_dir}/calib_${pft_id}
    echo "mpiexec -n 16 python calibrate_emulate_sample.py --pft ${pft_id} --bootstraps 100" >> ${job_dir}/calib_${pft_id}
    qsub ${job_dir}/calib_${pft_id}
done
