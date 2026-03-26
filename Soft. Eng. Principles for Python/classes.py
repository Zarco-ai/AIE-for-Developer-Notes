from characters import Character, Warrior # This line lets you grab classes from a package

'''Classes'''
# Create two distinct objects (instances) from the Character class
warrior1 = Character(100, 50, 10) #By doing this you are creating an object with a starting amount for its attributes ('.health', '.damage', '.speed')
ninja = Character(80, 40, 40)

# Show initial speeds
print(f"Warrior speed: {warrior1.speed}") #10
print(f"Ninja speed: {ninja.speed}")#40
# Apply the method only to the warrior instance
warrior1.double_speed()
# Show speeds after the modification
print(f"Warrior speed: {warrior1.speed}") #20
print(f"Ninja speed: {ninja.speed}") #40
# Take damage method
print(f"N_Health before damage: {ninja.health}") #80
ninja.take_damage(40)
print(f"N_Health after damage: {ninja.health}")#40

'''Inheritance'''
warrior2 = Warrior(100, 50, 10)
print(f'Initial health: {warrior2.health}')
warrior2.take_damage(40)
print(f'Health after damage: {warrior2.health}')

help(Character) # It pulls up the docstrings (the text descriptions) written by the programmer who created the object. 
                # It tells you what parameters to pass and what the function actually returns.
dir(Character)  # It returns a simple list of all the attributes and methods available to that object.
