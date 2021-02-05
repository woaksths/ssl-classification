# import lexicon_utils

NEGATOR = ["aint", "arent","cannot","cant","couldnt","darent", "didnt","doesnt","ain't","aren't",
        "can't","couldn't","daren't","didn't", "doesn't","dont","hadnt", "hasnt","havent","isnt",
        "mightnt","mustnt","neither","don't","hadn't","hasn't","haven't","isn't",
        "mightn't","mustn't","neednt","needn't","never","none","nope","nor",
        "not","nothing","nowhere","oughtnt","shant","shouldnt","uhuh","wasnt",
        "werent","oughtn't","shan't","shouldn't","uh-uh", "wasn't","weren't","without",
        "wont","wouldnt","won't","wouldn't","rarely","seldom","despite","no"]

CONTRAST = ['but', 'however', 'But', 'However']

END_WORDS = ['.', ',', '!', '?', '...',';', "'", '-', ')', '"', "because", "therefore", "so"]

STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 'one','two', 'three', 'four', 'first',
			'should', 'now', '>', 'br', '<', 'still', 'even', 'bit', 'people'
              ,'wednesday', 'yesterday', 'monday','friday', 'sunday', 'tuesday', 'thursday', 'saturday', '[SEP]' ,'[UNK]']

FILTER_WORDS= ['is','was','were','have', 'do', 'are', 'get', 'make', 'got', 'has', 'had','made', 'got', 'i', 'people']

