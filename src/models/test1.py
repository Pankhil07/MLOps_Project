from src.data.data import  test_set

test_set1= test_set
english_sentences = []
for i in test_set1:
    english_sentences.append(i['translation'].get('en'))


print(type(english_sentences))