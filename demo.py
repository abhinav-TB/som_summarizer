from som_summarizer import summarizer

input = '''(CNN) -- As Barack Obama makes his case to the nation for taking the fight to ISIS, his top diplomat is also trying to make sure America doesn't have to go it alone. U.S. Secretary of State John Kerry is sweeping through the Middle East to try to convince regional leaders to back America's plan to beat back the terror group, which has seized a large chunk of territory stretching from northern Syria to central Iraq with alarming pace in recent months. So who's with them? .'''
s = summarizer(epochs=100 ) # epochs #size of map
print(s.generate_summary(input))