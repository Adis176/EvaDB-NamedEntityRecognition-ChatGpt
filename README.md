# EvaDB-NamedEntityRecognition-ChatGpt
* All the main code can be found in the **'execute.py'** file.
* The csv files present in the 'data' folder will all be needed too.
* **'test'.csv'** will act as our data source, currently 50 rows have been iterated over, their output present in _'ans.csv'_ file in _'data' _folder.
* The **requirements.txt** lists the pre-requisites needed in order to run the following file.

To execute the code ( exec.py file), 
1. Clone this repository
2. Make sure that all the data folder files, i.e. the csv files are present accordingly. Shift the paths accordingly
3. run pip install requirements.txt. Requirements include [evadb, tiktoken, openai, time, re, json, pandas]. All these commands are present as comments in the **'exec.py'** file. 
4. Set your openAI key accordingly to make the API calls.
5. The dataset used id CONLL2003 one, a standard dataset used for NER tasks.

* Some datasets and other necessary files are present in the folder **‘data’**, which should all be imported.
* In order to bypass the Chat-Gpt rate limitation, a manual delay of 20 seconds would be needed after each query execution. Problem was that the inbuilt chat-gpt function took an entire table each time.
* To circumvent this, I decided to parse a single row into a temporary csv file named ‘temp.csv’ each time, so that a table could be formed by selecting all the data from the ‘temp.csv’ file (i.e. the current row).
* This allows us to parse the dataset, one row at a time, while introducing a gap of 20 seconds after each API call.
* Moreover, at each stage/iteration, we try to separate the words labelled and obtained through the prediction of the Chat-Gpt model as per the different categories – Location, organization, person, and miscellaneous.
* Then these words are compared such that we get the true positive value – how many words actually match, false positive rate – entities which are counted in our predicted output, but are actually not present in the original labelling given in the dataset, and false negatives – entities which were present in the dataset given but are not identified in the predicted output. This all helps us to calculate the F1-score.
