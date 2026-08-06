made = []
for i in range(3):
    made.append(lambda i=i: i)
print([f() for f in made])
