# import nltk
# nltk.download('wordnet')

from nltk.corpus import wordnet as wn

for synset in wn.synsets('wipe_up'):
    print("{}:".format(synset.name()))
    print("\t definition: {}".format(synset.definition()))
    print(f"\t hypornyms: {synset.hyponyms()}")
    hypernyms = ", ".join([hypernym.name() for hypernym in synset.hypernyms()])
    print("\t hypernyms: {}".format(hypernyms))
    print()
