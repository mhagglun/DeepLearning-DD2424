import sys
import os
import json
import html
import unidecode
import re
import numpy as np


def read_tweets(filename, output):
    """Parses the given json file, removing any non latin-1 character, URLs 
    and adds padding to each line so that its 140 characters long.
    The result is then written in lower case to the path specified by the ouput argument.

    Arguments:
        filename {string} -- The path to the json file to parse
        output {string} -- The path to the file to write the output to
    """
    output_txt = ''
    with open(filename, 'r') as json_file:
        tweets = json.load(json_file)
        for tweet in tweets:
            # Read tweet, ignoring characters which are not part of latin-1
            txt = tweet['text'].encode('latin-1', 'ignore').decode('latin-1')
            # Replace special characters such as â with a
            txt = unidecode.unidecode(txt)
            # Remove html characters like &amp
            txt = html.unescape(txt)
            # remove urls
            txt = re.sub(r'http\S+', '', txt)
            # convert all letter to lower case
            txt = txt.lower()
            # Padd text so that its 140 characters
            txt = txt.ljust(140)
            output_txt += txt

    with open(output, 'w') as output_file:
        output_file.write(output_txt)


def parse_all_tweets(directory='./data/trump_tweet_data_archive/', output='data/raw_tweets.txt'):
    """Parses all json files in the specified directory using read_tweets() and writes the output to a .txt file

    Keyword Arguments:
        directory {str} -- [The path to the directory containing the json files] (default: {'./data/trump_tweet_data_archive-master/'})
        output {str} -- [The path to where the output file will be written to] (default: {'data/raw_tweets.txt'})
    """
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.json'):
            read_tweets(directory+filename, output)
