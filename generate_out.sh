# generates all tabular output for each csv file

cd src
# for each data file
for file in ../data/*
do
    echo "processing " $file
    # run main.py to generate tables, with the csv name.
    # saves to out/csv_name.out
    (time python3 main.py -f $file)  > "../out/$(basename $file .csv).out" 2>&1
done
cd ..