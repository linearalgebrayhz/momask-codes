mkdir -p /scratch/network/hy4522/Momask-data/checkpoints
cd /scratch/network/hy4522/Momask-data/checkpoints

mkdir t2m
cd t2m
wget "https://drive.google.com/file/d/1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0/view?usp=sharing" humanml3d_models.zip
unzip humanml3d_models.zip
rm humanml3d_models.zip
cd ..

mkdir kit
cd kit
wget "https://drive.google.com/file/d/1FapdHNkxPouasVM8MWgg1f6sd_4Lua2q/view?usp=sharing" kit_models.zip
unzip kit_models.zip
rm kit_models.zip
cd ../../..

ln -s /scratch/network/hy4522/Momask-data/checkpoints /home/hy4522/Research/momask-codes/checkpoints