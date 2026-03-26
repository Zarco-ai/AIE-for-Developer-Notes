'''Learning about classes, inheritance'''

class Character: #Be sure to create a docstring for each class, everytime
    # Initialize attributes for the specific instance
    def __init__(self, health, damage, speed):
        self.health = health
        self.damage = damage
        self.speed = speed

    # This method modifies the instance's speed attribute
    def double_speed(self): # All this means is that you need to call the instance to use this method
                            # with no other arguments, for now, sum like: "warrior.double_speed()" 
                            # , and it will double the speed(attribute) for that  character(object/instance)
        self.speed *= 2
        
    def take_damage(self, amount):
        self.health -= amount
        
class Warrior(Character): # This line right here creates a class that has inherited every attribute and method from the 'Character' class
    def __init__(self, health, damage, speed):  # This line actually gets rid of the attributes from the parent
        super().__init__(health, damage, speed) # This actually brings back the attributes and NOW allows us to create a new object that is similar to the parent class
                                             #### .'super()' refers to the current class's parent's class, and this is passing Character's __init__ method to pass the attributes in the current class
        self.toughness = 0.90   # Toughness-attribute for this specific character(Object/Instance)
        
    def take_damage(self, amount): # Creates a new method for this character(object/instance) called take_damage: it overrides the parent's functionality of the method with the same name, until it gets to 'super()'
        mod_amount = amount * self.toughness # If 'amount' is 40, answer is 36
        super().take_damage(mod_amount) # Here we take the functionailty of our parent class by calling 'super()' to refer to our parent class,
                                        # then we take the 'take_damage' method to get the functionality of changing the 'health' attribute of our warrior to 64