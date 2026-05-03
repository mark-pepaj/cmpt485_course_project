# cmpt485_course_project

This project aims to experiment with the generative capabilities of the Transformer architecture by training the model only on a large dataset of recipes and analyzing the models ablility to generate recipes.

We took a CSV file of recipes and extracted the title, ingredients, and directions.
Then we formatted each section and used special tokens to indicate the sections.
The format is as follows:

<SOS>
<TITLE>Recipe</TITLE>
  
<INGREDIENTS>
ingredient_1
ingredient_2
    ...
ingredient_n
</INGREDIENTS>

<DIRECTIONS>
direction_1
direction_1
    ...
direction_n
</DIRECTIONS>
<EOS>


Then a prompt is randomly chosen from a set of prompts and inserted before after the <SOS> token:


<SOS>
<PROMPT>Show me a [title] recipe</PROMPT>
  
<TITLE>Recipe</TITLE>
  
<INGREDIENTS>
ingredient_1
ingredient_2
    ...
ingredient_n
</INGREDIENTS>

<DIRECTIONS>
direction_1
direction_1
    ...
direction_n
</DIRECTIONS>
<EOS>





Adapted from Andrej Karpathy's nanoGPT repository: https://github.com/karpathy/nanoGPT.git
