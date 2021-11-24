import re
str = 'what is your name. Dhruv/Meduri'
words = re.split(r"[\s \n - _ / . ? ,]",str)
print(words)
