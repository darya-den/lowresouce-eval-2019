"""Building models for a language (with lemma database)"""
from math import log
import shelve

def vocab_model(file_name, model_db_name, inflex_model_db_name,
                lemma_model_db_name):
    """Build models from file.

    Model contains entries for each word with the info
    about each lemma, inflexion, pos and morphological tag
    that this word has with the probability of this combination
    of word and its characteristics.

    model_db_name = evn.model_train
    inflex_model_db_name = evn.model_inflex_train
    lemma_model_db_name = evn.model_lemma_train """
    word_count = 0
    word_dict = {}
    inflexion_count = 0
    inflexion_dict = {}
    lemma_count = 0
    lemma_dict = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        my_file = file.readlines()
    for nline, line in enumerate(my_file):
        # skip empty lines
        if line.strip():
            if line.startswith("#"):
                continue
            # parse line
            # and find relevant attributes
            line_segm = line.split("\t")
            index = line_segm[0]
            if "-" in index:
                text_word = line_segm[1].lower()
                next_line = my_file[nline+1]
                next_line_segm = next_line.split("\t")
                word = next_line_segm[1].lower()
                lemma = next_line_segm[2]
                pos = next_line_segm[3]
                if pos == "X":
                    continue
                if pos == "PROPN":
                    pos = "NOUN"
                morph_gram = "_"
                inflexion = "#"
                # find segment with morph tags
                for i, s in enumerate(next_line_segm):
                    if i > 3:
                        if not s == "_":
                            morph_gram = s.strip()
                            break
                # derive inflexion
                # as a difference between word and lemma
                for n in range(1, len(word)):
                    if word[:n] == lemma:
                        inflexion = word[n:]
                        break
                part_line = my_file[nline+2]
                part = part_line.split()[1]
                if inflexion == "#":
                    inflexion = part
                else:
                    inflexion = inflexion + part
                entry = "(" + pos + ", " + morph_gram + ")"
                word_count += 1
                inflexion_count += 1
                lemma_count += 1
                if morph_gram == "_":
                    add = 0.25
                else:
                    add = 1
                x = word_dict.setdefault(text_word, {})
                y = x.setdefault(lemma, {})
                z = y.setdefault(inflexion, {})
                z.setdefault(entry, 0)
                word_dict[text_word][lemma][inflexion][entry] += add
                q = inflexion_dict.setdefault(inflexion, {})
                q.setdefault(entry, 0)
                inflexion_dict[inflexion][entry] += add
                r = lemma_dict.setdefault(lemma, {})
                r.setdefault(entry, 0)
                lemma_dict[lemma][entry] += add
            else:
                word = line_segm[1].lower()
                lemma = line_segm[2]
                # skip line with no info about lemma
                if lemma == "UNKN":
                    continue
                pos = line_segm[3]
                if pos == "X":
                    continue
                if pos == "PROPN":
                    pos = "NOUN"
                #if pos == "PART":
                    #continue
                # default values
                # for zero inflexion and null morph tag
                morph_gram = "_"
                inflexion = "#"
                # find segment with morph tags
                for i, s in enumerate(line_segm):
                    if i > 3:
                        if not s == "_":
                            morph_gram = s.strip()
                            break
                # derive inflexion
                # as a difference between word and lemma
                for n in range(1, len(word)):
                    if word[:n] == lemma:
                        inflexion = word[n:]
                        break
                entry = "(" + pos + ", " + morph_gram + ")"
                word_count += 1
                inflexion_count += 1
                lemma_count += 1
                if morph_gram == "_":
                    add = 0.25
                else:
                    add = 1
                x = word_dict.setdefault(word, {})
                y = x.setdefault(lemma, {})
                z = y.setdefault(inflexion, {})
                z.setdefault(entry, 0)
                word_dict[word][lemma][inflexion][entry] += add
                q = inflexion_dict.setdefault(inflexion, {})
                q.setdefault(entry, 0)
                inflexion_dict[inflexion][entry] += add
                r = lemma_dict.setdefault(lemma, {})
                r.setdefault(entry, 0)
                lemma_dict[lemma][entry] += add
    # now that's the dictionary is created
    # we need to establish the probability of
    # word-lemma-inflexion-(pos, morph_gram) combination
    for word in word_dict:
        for lemma in word_dict[word]:
            for inf in word_dict[word][lemma]:
                for tag in word_dict[word][lemma][inf]:
                    tag_count = word_dict[word][lemma][inf][tag]
                    tag_count = -round(log((tag_count+1)/word_count), 2)
                    word_dict[word][lemma][inf][tag] = tag_count
    with shelve.open(model_db_name) as db:
        db.update(word_dict)
    # same for the inflexion dict
    for infl in inflexion_dict:
        for tag in inflexion_dict[infl]:
            tag_count = inflexion_dict[infl][tag]
            tag_count = -round(log((tag_count+1)/inflexion_count), 2)
            inflexion_dict[infl][tag] = tag_count
    print(inflexion_count)
    with shelve.open(inflex_model_db_name) as infl_db:
        infl_db.update(inflexion_dict)
    for lemma in lemma_dict:
        for tag in lemma_dict[lemma]:
            tag_count = lemma_dict[lemma][tag]
            tag_count = -round(log((tag_count+1)/lemma_count), 2)
            lemma_dict[lemma][tag] = tag_count
    with shelve.open(lemma_model_db_name) as lemma_db:
        lemma_db.update(lemma_dict)

def tag_model(file_name, tag_model_db_name):
    """Build a model for tag co-occurrence..

    Model contains probabilities of a pair of tags
    occurring sequentially in a sentence.

    tag_model_db_name = evn.model_tag_train"""
    tags = []
    tag_dic = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        my_file = file.readlines()
    for line in my_file:
        if line.strip():
            if line.startswith("#"):
                tags.append("START")
                continue
            line_segm = line.split("\t")
            word = line_segm[1].lower()
            lemma = line_segm[2]
            if lemma == "UNKN":
                continue
            pos = line_segm[3]
            if pos == "_":
                continue
            #if pos == "PART":
                #continue
            if pos == "X":
                tags.append("#")
                continue
            if pos == "PROPN":
                pos = "NOUN"
            morph_gram = "_" # default for no morph information
            for i, s in enumerate(line_segm):
                if i > 3:
                    if not s == "_":
                        morph_gram = s.strip()
                        break
            entry = "(" + pos + ", " + morph_gram + ")"
            tags.append(entry)
        else:
            tags.append("END")
    n_tags = 0
    for i, tag in enumerate(tags):
        if not tag == "END":
            try:
                n_tags += 1
                next_tag = tags[i+1]
                if not next_tag == "START":
                    x = tag_dic.setdefault(tag, {})
                    x.setdefault(next_tag, 0)
                    tag_dic[tag][next_tag] += 1
            except IndexError:
                break
    for tag in tag_dic:
        next_tags = tag_dic[tag]
        for next_tag in next_tags:
            tag_count = tag_dic[tag][next_tag]
            tag_count = -round(log((tag_count+1)/n_tags), 2)
            tag_dic[tag][next_tag] = tag_count
    with shelve.open(tag_model_db_name) as db:
        db.update(tag_dic)
