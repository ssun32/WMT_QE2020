git clone https://github.com/facebookresearch/mlqe.git
mv mlqe/data .
rm -rf mlqe
cd data
for f in *.tar.gz; 
do
    tar -xvzf $f
    rm $f
done
