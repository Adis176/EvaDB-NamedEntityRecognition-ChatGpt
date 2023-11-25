# pip install --quiet "evadb[document,notebook]"

import evadb
cursor = evadb.connect().cursor()

# DO NOT FORGET TO ENTER YOUR OPENAI AI KEY HERE !!
import os
os.environ['OPENAI_API_KEY'] = 'sk-ENTER_YOUR_KEY_HERE'
open_ai_key = os.environ.get('OPENAI_API_KEY')

# !pip install eva--decord


# WARNING: 
# OPENAI OR RATHER TIKTOKEN MODULEMIGHT NOT WORK WITH SOME OPENAI VERSIONS. 
# IN SUCH CASES, IT WOULD BE BETTER IF A SPECIFIC VERSION OF OPENAI - 0.27.0 IS DOWNLOADED
# FOR THAT FIRST UNINSTALL THE CURRENT OPENAI VERSION, IF ANY PRESENT
# THEN DO INSTALL THE SPECIFIC OPENAI VERSION
# RESTART THE TERMINAL FOR THE CHANGES TO TAKE PLACE

# !pip uninstall openai
# !pip install openai==0.27.0

# !pip install tiktoken


import pandas as pd
import time
import json
import re
import tiktoken



total_cost = 0
cost_per_token = 0.000004


df = pd.read_csv('test.csv')


def count_tokens(prompt):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(prompt))
    return num_tokens


def convert_dict_values_to_lower(dict_obj):
  lower_dict = {}
  for key, value in dict_obj.items():
    if isinstance(value, str):
      lower_dict[key] = value.lower()
    elif isinstance(value, list):
      lower_dict[key] = [item.lower() for item in value]
    elif isinstance(value, dict):
      lower_dict[key] = convert_dict_values_to_lower(value)
    else:
      lower_dict[key] = value
  return lower_dict

def convert_list_to_lower(list_obj):
  lower_list = [item.lower() for item in list_obj]
  return lower_list


def process_batch(batch):
  for i in batch.index:
    processed_df = pd.DataFrame([{'example': batch['example'][i]}], columns=['example'])
    curr_tokens = count_tokens(batch['example'][i])

    # 138 tokens are in the prompt being passed.
    total_tokens = curr_tokens + 138

    # here, in order to match all words as per their category, we get them seprarately and convert all such words to lowercase
    data_dict = json.loads(batch['tokens'][i])
    data_dict = convert_dict_values_to_lower(data_dict)
    
    # getting each given labelled category words in different arrays for comparison later
    loc_words = data_dict["Location"]
    org_words = data_dict["Organization"]
    per_words = data_dict["Person"]
    mis_words = data_dict["Miscellaneous"]
    
    # Here, we append the current row to a csv file temporarily, so it can be later read by the Chat-gpt function and parsed as the current input
    processed_df.to_csv('temp.csv', index=False, header=True)
    temp_df = pd.read_csv('temp.csv')
   
   # we like create a new table, and put the current row in it, from the csv file 'temp'
    cursor.query("""
      DROP TABLE IF EXISTS Temporary;
    """).df()

    cursor.query("""
        CREATE TABLE IF NOT EXISTS Temporary (
            example TEXT(5000)
        );
    """).df()

    cursor.query("LOAD CSV 'temp.csv' INTO Temporary").df()

    # Here, we pass the current row into chat-gpt with a prompt in order to extract all the required entities from it.
    chatgpt_udf = """
        SELECT ChatGPT( "prompt: 'You are an expert at extracting Person, Organization, Location, and Miscellaneous entities from text. Your job is to extract named entities mentioned in text, and classify them into one of the following categories:',
            'labels': [
                'Location',
                'Organization',
                'Person',
                'Miscellaneous'
            ],
            For the given example,
            Example input - 'The role of the 70,000 mainly Kurdish village guards who fight Kurdistan Workers Party ( PKK ) guerrillas in the southeast has been questioned recently after media allegations that many of them are involved in common crime .'
            Output - '{Location: [], Organization: [Kurdistan Workers Party, PKK], Person: [], Miscellaneous: [Kurdish]}'
        }", example)
        FROM Temporary;
    """
    ans = cursor.query(chatgpt_udf).df()

    # We also add the answer token in the total count, to compute what is the actual cost. 
    total_tokens += count_tokens(ans['response'][0])
    curr_cost = total_tokens * cost_per_token


    # Now, the answer is in a string format, we try to separate each word as per the category for comparison
    pattern1 = r'Location: \[(.*?)\]'
    match1 = re.search(pattern1, ans['response'][0])
    if match1:
        output1 = match1.group(1).split(', ')
        location_words = [element.strip('"') for element in output1]
    else:
        location_words = []

    pattern2 = r'Organization: \[(.*?)\]'
    match2 = re.search(pattern2, ans['response'][0])
    if match2:
        output2 = match2.group(1).split(', ')
        organization_words = [element.strip('"') for element in output2]
    else:
        organization_words = []

    pattern3 = r'Person: \[(.*?)\]'
    match3 = re.search(pattern3, ans['response'][0])
    if match3:
        output3 = match3.group(1).split(', ')
        person_words = [element.strip('"') for element in output3]
    else:
        person_words = []

    pattern4 = r'Miscellaneous: \[(.*?)\]'
    match4 = re.search(pattern4, ans['response'][0])
    if match4:
        output4 = match4.group(1).split(', ')
        misc_words = [element.strip('"') for element in output4]
    else:
        misc_words = []

    # We convert each word to lower-case
    location_words = convert_list_to_lower(location_words)
    organization_words = convert_list_to_lower(organization_words)
    person_words = convert_list_to_lower(person_words)
    misc_words = convert_list_to_lower(misc_words)

    # for calculating the F1-score, we get the true positives, false positives and false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # we construct sets of each particular category, comparing the given labelled words with the predicted/obtained
    # The words which are in common denote the true positives
    common_words1 = set(loc_words) & set(location_words)
    common_words2 = set(org_words) & set(organization_words)
    common_words3 = set(per_words) & set(person_words)
    common_words4 = set(mis_words) & set(misc_words)
    true_positives += len(common_words1)
    true_positives += len(common_words2)
    true_positives += len(common_words3)
    true_positives += len(common_words4)

    # Here, un1 will denote the false positives. i.e. the words which were actually present in the given dataset, but not in the predicted outcome
    # While un2 willdenote the false negatives, i.e. the words obtained in the predicted outcome but are not actually prsent in the given dataset
    un1 = len(loc_words) - len(common_words1) + len(org_words) - len(common_words2) + len(per_words) - len(common_words3) + len(mis_words) - len(common_words4)
    un2 = len(location_words) - len(common_words1) + len(organization_words) - len(common_words2) + len(person_words) - len(common_words3) + len(misc_words) - len(common_words4)
    false_positives += un1
    false_negatives += un2

    # Calculating F1 score 
    if true_positives==0:
      precision=0
      recall=0
    else:
      precision = true_positives / (true_positives + false_positives)
      recall = true_positives / (true_positives + false_negatives)

    if precision==0 or recall==0:
      f1_score=0
    else:
      f1_score = 2 * (precision * recall) / (precision + recall)
    # print("F1: ", f1_score, "\n")

    # We append all results to a csv file, so it cna be easily exported and observed later.
    to_be_appended = pd.DataFrame([{'f1_score': f1_score, 'cost': curr_cost, 'output': ans['response'][0]}])
    print(to_be_appended)
    to_be_appended.to_csv('ans.csv', mode='a', header=False)

    # A delay of 20 seconds has to be introduced between each query, as chat-gpt gets rate-limited if more than 3 queries are passed to the API in less than a minute. 
    time.sleep(20)
    # print("Delay")

# Here, we try to iterate over the whole given dataset, row by row.
for i in range(0, len(df), 3):
  batch = df.iloc[i:i+3]
  process_batch(batch)
