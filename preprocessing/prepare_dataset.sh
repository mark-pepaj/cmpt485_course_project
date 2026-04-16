#! /usr/bin/bash

if ! [ -e "training.txt" ] && ! [ -e "validation.txt" ] && ! [ -e "testing.txt" ]
then
    echo "Downloading .zip file..."
    # download the .zip file
    wget -q --show-progress "https://www.kaggle.com/api/v1/datasets/download/wilmerarltstrmberg/recipe-dataset-over-2m"
    echo "Done"

    echo "Unzipping..."
    # extract the files from the zip
    unzip -q "recipe-dataset-over-2m"
    echo "Done"

    rm "recipe-dataset-over-2m"

    echo "Extracting data from recipes_data.csv..."
    # extract the columns we need (title, ingredients, directions)
    # create two extra instances of each row
    head -1 recipes_data.csv > recipes.csv && tail -n +2 recipes_data.csv | shuf >> recipes.csv
    python extract_cols.py  
    echo "Done"

    rm "recipes_data.csv"

    echo "Shuffling the data..."
    # shuffle the csv
    head -1 recipes.csv > shuffled.csv && tail -n +2 recipes.csv | shuf >> shuffled.csv
    echo "Done"

    rm "recipes.csv"
    
    # parse data, normalizing leading or trailing whitespaces
    # the ingredients and directions also are in a string list format initially, so we parse the ingredients and directions and convert them to a string
    python parse_data.py

    rm "shuffled.csv"
        
    # create the format for each sample
    # and create the training, validation, and testing set files with the respective splits 0.2, 0.4, 0.4
    python build_data_splits.py 

    rm "normalized.csv"

    echo "All done."
fi
