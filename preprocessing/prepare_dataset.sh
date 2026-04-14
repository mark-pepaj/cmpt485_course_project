#! /usr/bin/bash

if ! [ -e "training.txt" ] && [ -e "validation.txt" ] && [ -e "testing.txt" ]
then
    echo "Downloading .zip file..."
    # download the .zip file
    wget -q --show-progress "https://www.kaggle.com/api/v1/datasets/download/wilmerarltstrmberg/recipe-dataset-over-2m"
    echo "Done"

    echo "Unzipping..."
    # extract the files from the zip
    unzip -q "recipe-dataset-over-2m"
    echo "Done"

    echo "Extracting data from recipes_data.csv..."
    # extract the columns we need (title, ingredients, directions)
    # create two extra instances of each row
    python extract_cols.py  
    echo "Done"

    echo "Shuffling the data..."
    # shuffle the csv
    shuf recipes.csv > shuffled_recipes.csv
    echo "Done"

    # remove these files since we're done processing them
    rm "recipe-dataset-over-2m"
    rm "recipes_data.csv"
    rm "recipes.csv"

    # create the format for each sample
    #python format_samples.py

    # create the training, validation, and testing set files with the respective splits 0.2, 0.4, 0.4
    #python build_data_splits.py
fi
