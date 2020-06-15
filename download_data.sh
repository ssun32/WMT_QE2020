git clone https://github.com/facebookresearch/mlqe.git
mv mlqe/data .
rm -rf mlqe
cd data
wget https://www.quest.dcs.shef.ac.uk/wmt20_files_qe/ru-en.tar.gz
wget https://www.quest.dcs.shef.ac.uk/wmt20_files_qe/ru-en_test.tar.gz
for f in *.tar.gz; 
do
    tar -xvzf $f
    rm $f
done
cd ../

python sample_data.py
