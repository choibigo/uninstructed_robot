from Faker import faker

randd = faker('ko-KR')
data = [(randd.name(), randd.pyint(min_value=13, max_value=18) for i in range(20))]

