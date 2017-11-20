import json
import re
import pandas

from nltk.corpus import stopwords
from pytrends.request import TrendReq

if __name__ == '__main__':
    # Open up pre-processed location corpus
    with open('english_sms_corpus.json', 'r') as fp:
        obj = json.load(fp)

    # Access Google Trends API
    pytrends = TrendReq(hl='en-US', tz=360)
    stop_words = set(stopwords.words("english"))

    for key, value in obj.items():
        # Just get an example to play with
        if value == "united kingdom":
            message = key.strip().lower()
            message = re.sub(r"[^A_Za-z0-9\s]", "", message)
            message = re.sub(r"\s[0-9]\s", " ", message)
            message = message.split()
            # Remove stop words
            message2 = []
            for word in message:
                if word not in stop_words:
                    message2.append(word)
            message = message2
            print message

            # Query the API
            for word in message:
                kw_list = [word]
                pytrends.build_payload(kw_list, cat=0, timeframe="today 5-y",
                                       geo='', gprop='')
                loc_table = pytrends.interest_by_region(resolution='COUNTRY')
                # Get counts for query by each country (100 is top)
                uk = loc_table.loc["United Kingdom", word]
                us = loc_table.loc["United States", word]
                sg = loc_table.loc["Singapore", word]

            break   # So we stop after one example we played with
