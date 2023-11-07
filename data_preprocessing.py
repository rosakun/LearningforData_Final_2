import emoji
import re


def emoji_to_text(input_text):
    # Use the emoji.emojize() function to convert emojis to textual format
    text_with_emojis = emoji.demojize(input_text)
    return text_with_emojis

import re


def replace_multiple_user_strings(text):
    # Define a regular expression pattern to match '@USER' repeated more than two times in a row
    pattern = r'(@USER\s+){2,}'
    
    # Use re.sub to replace the matched pattern with '@USER @USER @USER '
    modified_text = re.sub(pattern, '@USER @USER ', text)
    
    return modified_text


def remove_url_occurrences(input_string):
    return input_string.replace('URL', '')



def preprocess_data(infile, outfile):

    # Reads in original data and applies various preprocessing steps.
    with open(infile, 'r', encoding='utf-8') as f: 
        with open(outfile, 'w', encoding='utf-8') as w:
            for line in f: 
                tweet = ' '.join(line.split()[:-1])
                label = line.split()[-1]

                # Replace occurences of '@USER' occuring over 3 times in a row with @USER @USER @USER
                tweet = replace_multiple_user_strings(tweet)

                # Remove occurences of 'URL'
                tweet = remove_url_occurrences(tweet)

                # Convert tweet to lowercase
                tweet = tweet.casefold()

                # Convert emojis to text format
                tweet = emoji_to_text(tweet)

                w.write(tweet + '\t' + label + '\n')



            

preprocess_data('data/test.tsv','data/test_preprocessed.tsv')