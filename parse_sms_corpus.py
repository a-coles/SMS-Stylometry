import json
# This script composes an English-language SMS corpus from the NUS SMS corpus
# and the Tag British SMS Corpus.

if __name__ == '__main__':
    with open('smsCorpus_en_2015.03.09_all.json', 'r') as fp:
        obj = json.load(fp)

    export_dict = {} # {message : country}

    for key, value in obj.items():
        for field, info in value.items():
            if field != "message":
                continue
            if info == "2015.03.09":
                continue
            for item in info:
                text = item['text']['$']
                country =  item['source']['userProfile']['country']['$'].lower()
                if country == "usa":
                    country = "united states"
                if country == "sg":
                    country = "singapore"
                if country == "uk":
                    country = "united kingdom"

                if country in ["united states", "united kingdom", "singapore"]:
                    export_dict[text] = country

    with open('british_sms_corpus.txt', 'r') as fp:
        british_messages = fp.readlines()

    for message in british_messages:
        export_dict[message] = "united kingdom"

    with open('english_sms_corpus.json', 'w') as fp:
        json.dump(export_dict, fp)
