import nltk
# nltk.download('wordnet')

#wordnet이 corpus안에 들어있으므로 불러오기
from nltk.corpus import wordnet

#syns라고 정의하여 program과 관련된 단어들을 찾도록 하기.synsets이 관련 단어들을 뽑아내 줌.
syns = wordnet.synsets("program")

# [Synset('plan.n.01'), Synset('program.n.02'), Synset('broadcast.n.02'), Synset('platform.n.02'), Synset('program.n.05'), Synset('course_of_study.n.01'), Synset('program.n.07'), Synset('program.n.08'), Synset('program.v.01'), Synset('program.v.02')]
print(syns)
# Synset('plan.n.01')
print(syns[0])
# [Lemma('plan.n.01.plan'), Lemma('plan.n.01.program'), Lemma('plan.n.01.programme')]
print(syns[0].lemmas())
# plan.n.01
print(syns[0].name())
# plan
print(syns[0].lemmas()[0].name())

# a series of steps to be carried out or goals to be accomplished
print(syns[0].definition())
# ['they drew up a six-step plan', 'they discussed plans for a new bond issue']
print(syns[0].examples())



synonyms = []
antonyms = []

for syn in wordnet.synsets("sweep"):
    for l in syn.lemmas():
        print("l:",l)
        synonyms.append(l.name())
        p=l.name()
        print('--------------',p)
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
            
print(set(synonyms))
print(set(antonyms))


w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cat.n.01")
print(w1.wup_similarity(w2))