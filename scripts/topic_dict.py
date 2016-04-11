topics = \
    [
    'Technology',
    'Science',
    'Books',
    'Business',
    'Movies',
    'Visiting and Travel',
    'Music',
    'Food',
    'Health',
    'Education',
    'Design',
    'Entertainment',
    'Cooking',
    'Economics',
    'Psychology',
    'History',
    'Writing',
    'Dating and Relationships',
    'Sports',
    'Photography',
    'Philosophy',
    'Finance',
    'Marketing',
    'Literature',
    'Fashion and Style',
    'Mathematics',
    'Politics',
    'Television Series',
    'Entrepreneurship',
    'Fine Art',
    'Facebook',
    'Computer Science',
    'Business Strategy',
    'Startups',
    'Technology Trends',
    'Physics',
    'Google',
    'Barack Obama',
    'Journalism',
    'Book Recommendations',
    'Nutrition',
    'YouTube',
    'Love',
    'Investing',
    'Healthy Eating',
    'Web Design',
    'Medicine and Healthcare',
    'Startup Founders and Entrepreneurs',
    'Startup Advice and Strategy',
    'Recipes',
    'Harry Potter',
    'Hollywood',
    'Television',
    'Lean Startups',
    'Reading',
    'India',
    'Business Models',
    'Scientific Research',
    'User Interfaces',
    'Novels',
    'Restaurants',
    'Musicians',
    'Writers and Authors',
    'User Experience',
    'The Big Bang Theory',
    'iTunes',
    'International Travel',
    'Life',
    'How I Met Your Mother',
    'Human Behavior',
    'The Universe',
    'Philosophy of Everyday Life',
    'Fiction',
    'Vacations',
    'Lady Gaga',
    'Software Engineering',
    'Money',
    'Exercise',
    'Baking',
    'Digital Photography',
    'Rock Music',
    'Web Marketing',
    'Literary Fiction',
    'Clothing and Apparel',
    'Amazon.com',
    'Hotels',
    'Social Psychology',
    'Social Media Marketing',
    'Computer Programming',
    'Inception',
    'Product Design',
    'Biology',
    'Cameras',
    'Small Businesses',
    'National Basketball Association',
    'Higher Education',
    'Social Advice',
    'Friendship',
    'Mental Health']


topic_clean = {}

def chk_model(b):
    d = []
    for bb in b:
        try:
            if bb in model:
                pass
        except:
            continue
        d.append(bb)
    return d

for j in topics:
    a = clean_ques(j)
    c = chk_model(a)
    topic_clean[j] = c



clus_topic = {}
simil_topic = {}

for i in data:
    d = []
    b = clean_ques(i)
    d = chk_model(d)
    if len(d) == 0:
        # print ("no clus")
        # print (i)
        # print (data[i], '\n')
        continue
    dist = -1
    try:
        print (d)
        simil = model.most_similar(positive=d)
        if (simil[0][1] > 0.55):
            simil_topic[i] = simil[0][0]
    except:
        print (sys.exc_info()[0], d)
        pass
    for j in topic_clean:
        try:
            if (len(topic_clean[j]) == 0):
                continue
            n = model.n_similarity(topic_clean[j], d)
        except:
            # print ("outer loop",sys.exc_info()[0], topic_clean[j], j, d)
            continue
        # print (n)
        n = abs(n)
        if (n > dist and n > 0.55):
            dist = n
            clus_topic[i] = j
    # print (i)
    # print (data[i])
    # if (i in clus_topic):
    #     print (clus_topic[i])
for i in data:
    try:
        print (i)
        print (data[i])
        if (i in clus_topic):
            print ("Clus got: ", clus_topic[i], "\n")
        if (i in simil_topic):
            print ("Simil got: ", simil_topic[i], "\n")
    except:
        pass