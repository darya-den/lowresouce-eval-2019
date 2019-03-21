"""Morphological tagging using models."""
import shelve

class Tagger(object):
    """Class for morphological tagger."""
    def __init__(self, model_db_name, inflexion_db_name, tag_db_name):
        """Initialize Tagger.

        param: model_db_name: path to the model_db.
                              model_db is {word: {lemma: {inflexion: {pos_morph: probability}}}}.
               inflexion_db_name: path to the inflexion_db.
                                  inflexion_db is {inflexion: {pos_morph: probability}}.
               tag_db_name: path to the tag_db.
                            tag_db is {pos_morph_tag: {following_pos_morph_tag: probability}}."""
        self.model_db = shelve.open(model_db_name)
        self.inflexion_db = shelve.open(inflexion_db_name)
        self.tag_db = shelve.open(tag_db_name)

    def __del__(self):
        self.model_db.close()
        self.inflexion_db.close()
        self.tag_db.close()
        
    def parse_sentence(self, file_name):
        """Parse the input file into sentences.

        param: file_name: path to the train file.
        return: list of sentences of the file."""
        sentences = []
        with open(file_name, 'r', encoding='utf-8') as f:
            new_sentence = []
            for line in f:
                if line.strip():
                    for sp in line.split():
                        if not sp == "_":
                            word = sp
                            break
                    new_sentence.append(word)
                else:
                    sentences.append(new_sentence)
                    new_sentence = []
            sentences.append(new_sentence)
        return sentences

    def get_tags(self, word):
        """Get all possible tags for word from dbs.

        If the word was in the train file (so it is in the model_db),
        get the tags from the model_db.
        If the word wasn't in the train file, use inflexion_db:
        first assume that the word has zero inflexion (#) and get the tags of #,
        then search for the inflexion in the end of the word.
        If a substring is in the inflex_db, get the tags of that substring.

        param: word: input word
        return: dictionary {lemma (or hypothetical lemma): {inflexion (hypot. inflexion): {pos_morph: prob}}}"""
        word = word.lower()
        tag_dict = {}
        # word was encountered in the train file
        if word in self.model_db:
            tag_dict = self.model_db[word]
        else:
            # get tags of zero inflexion
            x = tag_dict.setdefault(word, {})
            y = x.setdefault("#", {})
            tag_dict[word]["#"] = self.inflexion_db["#"]
            # check if a substring at the end of the word is an inflexion
            for i in range(1, len(word)):
                root = word[:i]
                inflexion = word[i:]
                if inflexion in self.inflexion_db:
                    tag_dict.setdefault(root, {})
                    tag_dict[root][inflexion] = self.inflexion_db[inflexion]
        return tag_dict

    def get_best_model(self, tags):
        """Get the most probable model of tags.

        First choose the best combination of tags for the first two words.
        Then choose the best i+1th tag based on the ith tag.
        
        param: tags: list of all possible tags for each word in a sentence.
        return: output_tags: list of dictionaries like
                             {'lemma': best_lemma, 'pos_morph': best_pos_morph, 'score': best_score}
                no_score_tags: list of dictionaries like
                               {'lemma': best_lemma, 'pos_morph': best_pos_morph}."""
        output_tags = []
        no_score_tags = []
        for i, tag in enumerate(tags):
            try:
                next_tag = tags[i+1]
            # reached the end of the sentence (last tag)
            except IndexError:
                if i == 0:
                    # there is just one word in a sentence
                    # just choose the most probable tag
                    best_score = 15
                    for lemma in tag:
                        for infl in tag[lemma]:
                            for pos_morph in tag[lemma][infl]:
                                score = tag[lemma][infl][pos_morph]
                                if score < best_score:
                                    best_score = score
                                    best_lemma = lemma
                                    best_tag = pos_morph
                    output_tags.append({'lemma': best_lemma, 'pos_morph': best_tag,
                                        'score': best_score})
                    no_score_tags.append({'lemma': best_lemma, 'pos_morph': best_tag})
                    return output_tags, no_score_tags
                else:
                    break
            if not i == 0:
                # we already know the best tag for the ith element
                # now we choose best tag for the i+1th element 
                best_tag = output_tags[i]
                score = best_tag['score']
                pos_morph = best_tag['pos_morph']
                try:
                    next_tags = self.tag_db[pos_morph]
                # the tag doesn't have any distributional info
                # i.e. we don't know what tag follow this one
                # then for the i+1th element we choose the best tag
                except KeyError:
                    best_score = 15
                    for lemma in next_tag:
                        for infl in next_tag[lemma]:
                            for pos_morph in next_tag[lemma][infl]:
                                score = next_tag[lemma][infl][pos_morph]
                                if score < best_score:
                                    best_score = score
                                    best_lemma = lemma
                                    best_tag = pos_morph
                    output_tags.append({'lemma': best_lemma, 'pos_morph': best_tag,
                                        'score': best_score})
                    no_score_tags.append({'lemma': best_lemma, 'pos_morph': best_tag})
                    continue
                # here ith tag has info about its distribution
                best_score = 15 # random large number so we can choose the best tag
                for lemma2 in next_tag:
                    lemmas2 = next_tag[lemma2]
                    for infl2 in lemmas2:
                        inflexs2 = lemmas2[infl2]
                        for pos_morph2 in inflexs2:
                            if pos_morph2 in next_tags:
                                cond_prob = next_tags[pos_morph2] - score
                            else:
                                cond_prob = 10.2 - score
                            if cond_prob < best_score:
                                best_score = cond_prob
                                next_lemma = lemma2
                                next_tag_best = pos_morph2
                                next_score - inflexs2[pos_morph2]
                output_tags.append({'lemma': next_lemma, 'pos_morph': next_tag_best, 'score': next_score})
                no_score_tags.append({'lemma': next_lemma, 'pos_morph': next_tag_best})
            else:
                # for the first element
                # find the most probable combination of the first tag
                # and the next tag
                best_score = 15
                for lemma1 in tag:
                    lemmas1 = tag[lemma1]
                    for infl1 in lemmas1:
                        inflexs1 = lemmas1[infl1]
                        for pos_morph1 in inflexs1:
                            tag1_score = inflexs1[pos_morph1]
                            try:
                                next_tags = self.tag_db[pos_morph1]
                            except KeyError:
                                print(pos_morph1)
                                continue
                            for lemma2 in next_tag:
                                lemmas2 = next_tag[lemma2]
                                for infl2 in lemmas2:
                                    inflexs2 = lemmas2[infl2]
                                    for pos_morph2 in inflexs2:
                                        if pos_morph2 in next_tags:
                                            cond_prob = next_tags[pos_morph2] - tag1_score
                                        else:
                                            cond_prob = 10.2 - tag1_score
                                        if cond_prob < best_score:
                                            best_score = cond_prob
                                            best_lemma = lemma1
                                            best_tag = pos_morph1
                                            tag_score = tag1_score
                                            next_score = inflexs2[pos_morph2]
                                            next_lemma = lemma2
                                            next_tag_best = pos_morph2
                output_tags.append({'lemma': best_lemma, 'pos_morph': best_tag, 'score': tag_score})
                output_tags.append({'lemma': next_lemma, 'pos_morph': next_tag_best, 'score': next_score})
                no_score_tags.append({'lemma': best_lemma, 'pos_morph': best_tag})
                no_score_tags.append({'lemma': next_lemma, 'pos_morph': next_tag_best})
        return output_tags, no_score_tags
                
    def tag_sentence(self, sentence):
        """Give best tags to the sentence.

        Get all tags for each word in the sentence,
        then get the best combination of tags.
        param: sentence: input sentence.
        return: best models"""
        tags = []
        for word in sentence:
            tags.append(self.get_tags(word))
        #print(sentence, tags)
        best_model = self.get_best_model(tags)
        return best_model

    def tag(self, file_name):
        """Tag input file.

        First parse the file into sentences,
        then tag each sentence.
        param: file_name: path to the file
        return: list of tagged sentences"""
        tagged_sentences = []
        no_score_tagged = []
        sentences = self.parse_sentence(file_name)
        for sentence in sentences:
            res = self.tag_sentence(sentence)
            tagged_sentences.append(res[0])
            no_score_tagged.append(res[1])
        return tagged_sentences, no_score_tagged
