# cmpt485_course_project

This project aims to experiment with the generative capabilities of the Transformer architecture by training the model only on a large dataset of recipes and analyzing the models ablility to generate recipes.

We took a CSV file of recipes and extracted the title, ingredients, and directions.
Then we formatted each section and used special tokens to indicate the sections.
A prompt is randomly chosen from a set of prompts and inserted into the recipe before the title.







Adapted from Andrej Karpathy's nanoGPT repository: https://github.com/karpathy/nanoGPT.git
