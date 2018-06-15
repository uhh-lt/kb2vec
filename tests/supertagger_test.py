from linkers.supertagger import SuperTagger


st = SuperTagger()
r = st._entity_link("Michael Jeffrey Jordan, also known by his initials, MJ, is an American former professional basketball player. He played 15 seasons in the National Basketball Association for the Chicago Bulls and Washington Wizards.")
for tag in r:
    print(tag, "\n")