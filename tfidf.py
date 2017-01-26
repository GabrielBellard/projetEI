from sklearn.feature_extraction.text import TfidfVectorizer

tfidfVect = TfidfVectorizer()
tfidf = tfidfVect.fit_transform(['trente', 'an', 'batail', 'Endor' ,'chute', 'Empire', 'galact'], ['souris', 'verte', 'courait', 'dans' ,'herbe', 'an', 'test'])
print tfidf