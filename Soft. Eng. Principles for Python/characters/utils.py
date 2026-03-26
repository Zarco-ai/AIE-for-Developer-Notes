'''Description/Purpose of this file:

This folder acts as a "junk drawer", holding onto our reusable, generic helper functions
that don't fit to well with the main logic of our primary classes. It:
    - DRY (Dont Repeat Yourself): If you find yourself writing the same bit of code 
    (like a date formatter or a regex cleaner) in three different files, you move it to utils.py so every other file can just import it.
    - Organization: It separates "low-level" logic (like file handling or string manipulation) from "high-level" logic (like your Character class or data processing).
    - Readability: By moving 50 lines of helper code into a utility file, your main script becomes much shorter and easier for someone else to read.
    
Example of what working with a utils file looks like is:

    from .token_utils import tokenize

    class Document:
        def __init__(self, text):
            # The complex work is hidden away in the utility file!
            self.tokens = tokenize(text)
        
Where 'tokenize' is a function created in the 'token_utils.py' file, which is also in the 'characters' folder/package,
that gathers the amount of tokens in a text with math and some low level logic.

Remember when importing functions from a utils file, or any file inside of your package folder, to write a '.' before the name
of the file you are importing the function from.

'''